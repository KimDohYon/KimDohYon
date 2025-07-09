#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <sstream>
#include <cstring>
#include "ap_int.h"

#define CH_NM 16
const int maxWords = (1 << 20) + 100;
ap_uint<512> inputData[maxWords];
ap_uint<512> outputData[1 << 20];

extern "C" void sha1Kernel(ap_uint<512>* inputData, ap_uint<512>* outputData);

std::string get_sha1_string(ap_uint<512> hash) {
    std::ostringstream oss;
    for (int i = 0; i < 20; ++i) {
        unsigned char byte = hash.range(i * 8 + 7, i * 8);
        oss << std::hex << std::setw(2) << std::setfill('0') << (int)byte;
    }
    return oss.str();
}

std::string debug_hex_512(ap_uint<512> word) {
    std::ostringstream oss;
    for (int i = 0; i < 64; ++i) {
        unsigned char byte = word.range(i * 8 + 7, i * 8);
        oss << std::hex << std::setw(2) << std::setfill('0') << (int)byte;
    }
    return oss.str();
}

int main() {
    const char* msgs[2] = {"abcde", "fghij"};
    const int msgLens[2] = {static_cast<int>(strlen(msgs[0])), static_cast<int>(strlen(msgs[1]))};
    const int totalMsgs = CH_NM * 2;

    std::vector<std::string> msgList;

    ap_uint<512> config = 0;
    config.range(511, 448) = msgLens[0];
    config.range(447, 384) = totalMsgs;
    inputData[0] = config;

    for (int i = 0; i < totalMsgs; ++i) {
        const char* msg = msgs[i % 2];
        int len = msgLens[i % 2];
        msgList.push_back(msg);

        ap_uint<512> msgBlock = 0;
        for (int j = 0; j < len; ++j) {
            msgBlock.range(j * 8 + 7, j * 8) = (unsigned char)msg[j];
        }

        inputData[1 + i] = msgBlock;

    }

    for (int i = 1 + totalMsgs; i < maxWords; ++i) {
        inputData[i] = 0;
    }

    std::cout << "[INFO] Sending " << totalMsgs << " messages to " << CH_NM << " channels\n";

    sha1Kernel(inputData, outputData);

    const std::string expected_hashes[2] = {
        "03de6c570bfe24bfc328ccd7ca46b76eadaf4334",  // abcde
        "27c97e303a5d4a98562a9f76696ba1fec889fc02"   // fghij
    };

    std::cout << "\n[DEBUG] Compare Results:\n";

    for (int i = 0; i < totalMsgs; ++i) {
        const std::string& msg = msgList[i];
        ap_uint<512> h = outputData[i];
        std::string hashStr = get_sha1_string(h);
        std::string expected = expected_hashes[i % 2];

        std::cout << "[CH" << std::setw(2) << (i % CH_NM) << "][msg: " << msg << "] SHA1: " << hashStr;
        std::cout << (hashStr == expected ? "  OK" : "  MISMATCH") << std::endl;
    }

    return 0;
}
