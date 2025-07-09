#include <ap_int.h>
#include <hls_stream.h>
#include "sha1.hpp"
#include "kernel_config.hpp"


void loadInput(
    ap_uint<512>* inputData,
    hls::stream<ap_uint<32>> msgStrm[CH_NM],
    hls::stream<ap_uint<64>> msgLenStrm[CH_NM],
    hls::stream<bool> eMsgLenStrm[CH_NM]) {

    ap_uint<512> cfg = inputData[0];
    ap_uint<64> msgLen = cfg.range(511, 448);
    ap_uint<64> msgNum = cfg.range(447, 384);

    for (ap_uint<64> i = 0; i < msgNum; i++) {
        ap_uint<512> data = inputData[1 + i];
        int ch = i % CH_NM;

        msgLenStrm[ch].write(msgLen);
        eMsgLenStrm[ch].write(false);

        for (int j = 0; j < msgLen; j += 4) {
#pragma HLS PIPELINE II=1
            ap_uint<32> w = 0;
            for (int b = 0; b < 4; ++b) {
                if ((j + b) < msgLen) {
                    ap_uint<8> byte = data.range((j + b) * 8 + 7, (j + b) * 8);
                    w.range(8 * b + 7, 8 * b) = byte;
                }
            }
            msgStrm[ch].write(w);
        }
    }

    for (int i = 0; i < CH_NM; i++) {
        eMsgLenStrm[i].write(true);
    }
}


void sha1Hash(
    hls::stream<ap_uint<32>> msgStrm[CH_NM],
    hls::stream<ap_uint<64>> msgLenStrm[CH_NM],
    hls::stream<bool> eMsgLenStrm[CH_NM],
    hls::stream<ap_uint<160>> hshStrm[CH_NM],
    hls::stream<bool> eHshStrm[CH_NM]) {
#pragma HLS INLINE off
    for (int i = 0; i < CH_NM; i++) {
#pragma HLS UNROLL
        xf::security::sha1<32>(msgStrm[i], msgLenStrm[i], eMsgLenStrm[i], hshStrm[i], eHshStrm[i]);
    }
}

void writeOutput(
    hls::stream<ap_uint<160>> hshStrm[CH_NM],
    hls::stream<bool> eHshStrm[CH_NM],
    hls::stream<ap_uint<512>>& outStrm,
    hls::stream<unsigned int>& burstLenStrm) {

    ap_uint<CH_NM> doneMask = ~0;
    const unsigned int burstLen = BURST_LEN;
    unsigned int burstCount = 0;

    while (doneMask != 0) {
        for (int i = 0; i < CH_NM; i++) {
#pragma HLS PIPELINE II=1
            if (!eHshStrm[i].empty()) {
                bool e = eHshStrm[i].read();
                if (!e) {
                    if (!hshStrm[i].empty()) {
                        ap_uint<160> h = hshStrm[i].read();
                        ap_uint<512> out = 0;
                        out.range(159, 0) = h;
                        outStrm.write(out);
                        if (++burstCount == burstLen) {
                            burstLenStrm.write(burstLen);
                            burstCount = 0;
                        }
                    }
                } else {
                    doneMask[i] = 0;
                }
            }
        }
    }

    if (burstCount != 0) {
        burstLenStrm.write(burstCount);
    }

    burstLenStrm.write(0);
}

void outputWriteBack(hls::stream<ap_uint<512>>& outStrm,
                     hls::stream<unsigned int>& burstLenStrm,
                     ap_uint<512>* outputData) {
    unsigned int outIdx = 0;
    unsigned int len = burstLenStrm.read();
    while (len != 0) {
        for (unsigned int i = 0; i < len; i++) {
#pragma HLS PIPELINE II=1
            outputData[outIdx++] = outStrm.read();
        }
        len = burstLenStrm.read();
    }
}

extern "C" void sha1Kernel(ap_uint<512> inputData[(1 << 20) + 100], ap_uint<512> outputData[1 << 20]) {
#pragma HLS INTERFACE m_axi offset=slave bundle=gmem0_0 port=inputData latency=64 \
    num_read_outstanding=16 max_read_burst_length=64
#pragma HLS INTERFACE m_axi offset=slave bundle=gmem0_1 port=outputData latency=64 \
    num_write_outstanding=16 max_write_burst_length=64

#pragma HLS INTERFACE s_axilite port=inputData bundle=control
#pragma HLS INTERFACE s_axilite port=outputData bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS DATAFLOW

    const unsigned int channelNum = CH_NM;
    const unsigned int burstLen = BURST_LEN;
    const unsigned int fifobatch = 4;
    const unsigned int fifoDepth = burstLen * fifobatch;
    const unsigned int msgDepth = fifoDepth * (512 / 32 / CH_NM);

    hls::stream<ap_uint<32>> msgStrm[CH_NM];
    hls::stream<ap_uint<64>> msgLenStrm[CH_NM];
    hls::stream<bool> eMsgLenStrm[CH_NM];
    hls::stream<ap_uint<160>> hshStrm[CH_NM];
    hls::stream<bool> eHshStrm[CH_NM];
    hls::stream<ap_uint<512>> outStrm;
    hls::stream<unsigned int> burstLenStrm;

#pragma HLS STREAM variable=msgStrm depth=msgDepth
#pragma HLS STREAM variable=msgLenStrm depth=128
#pragma HLS STREAM variable=eMsgLenStrm depth=128
#pragma HLS STREAM variable=hshStrm depth=128
#pragma HLS STREAM variable=eHshStrm depth=128
#pragma HLS STREAM variable=outStrm depth=fifoDepth
#pragma HLS STREAM variable=burstLenStrm depth=fifobatch

#pragma HLS bind_storage variable=msgStrm type=fifo impl=bram
#pragma HLS bind_storage variable=msgLenStrm type=fifo impl=lutram
#pragma HLS bind_storage variable=eMsgLenStrm type=fifo impl=lutram
#pragma HLS bind_storage variable=hshStrm type=fifo impl=lutram
#pragma HLS bind_storage variable=eHshStrm type=fifo impl=lutram
#pragma HLS bind_storage variable=outStrm type=fifo impl=bram
#pragma HLS bind_storage variable=burstLenStrm type=fifo impl=lutram

    loadInput(inputData, msgStrm, msgLenStrm, eMsgLenStrm);
    sha1Hash(msgStrm, msgLenStrm, eMsgLenStrm, hshStrm, eHshStrm);
    writeOutput(hshStrm, eHshStrm, outStrm, burstLenStrm);
    outputWriteBack(outStrm, burstLenStrm, outputData);
}
