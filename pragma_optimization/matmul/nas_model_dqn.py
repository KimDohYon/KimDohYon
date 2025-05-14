import torch
import torch.nn.functional as F
import numpy as np
import os
from parse_power import (
    load_baseline,
    evaluate_score,
    update_final_results,
    parse_results,
    parse_power_report,
    append_results_with_metrics
)

# === 설정 ===
factors = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 768]
num_factors = len(factors)
num_pragmas = 4  # sum_unroll, norm_unroll 제거

# === 학습 가능한 알파 초기화 ===
if os.path.exists("alpha.pt"):
    alpha = torch.load("alpha.pt")
    alpha.requires_grad_(True)
else:
    alpha = torch.zeros(num_pragmas, num_factors, requires_grad=True)

optimizer = torch.optim.Adam([alpha], lr=0.1)

# === softmax로 선택 확률 분포 생성 후 샘플링 ===
probs_list = [F.softmax(alpha[i], dim=-1) for i in range(num_pragmas)]
selected_indices = [torch.multinomial(p, 1).item() for p in probs_list]
selected_factors = [factors[idx] for idx in selected_indices]

# === 선택된 프라그마를 기록 (쉘 스크립트에서 읽음) ===
with open("current_pragma.txt", "w") as f:
    f.write(",".join(map(str, selected_factors)))

# === 코드 생성 ===
with open("matmul_template.cpp", "r", encoding="utf-8") as f:
    template = f.read()

placeholders = [
    "__X_BUFF_PARTITION__",
    "__XS_BUFF_PARTITION__",
    "__W_BUFF_PARTITION__",
    "__WS_BUFF_PARTITION__"
]
for i, ph in enumerate(placeholders):
    template = template.replace(ph, str(selected_factors[i]))

with open("matmul.cpp", "w") as f:
    f.write(template)

# === HLS 및 Vivado 실행 ===
os.system("timeout 1200 vitis_hls -f run_hls.tcl > vitis_log.txt 2>&1")
os.system("timeout 1200 vivado -mode batch -source run_vivado_power.tcl > vivado_log.txt 2>&1")

# === 결과 파싱 ===
results_fields, interval = parse_results()
power = parse_power_report()
base_i, base_p = load_baseline()
score = evaluate_score(interval, power, base_i, base_p)

# === 학습 (REINFORCE-style) ===
log_prob = sum(torch.log(probs_list[i][selected_indices[i]] + 1e-8) for i in range(num_pragmas))
loss = -score * log_prob
optimizer.zero_grad()
loss.backward()
optimizer.step()

# === alpha 저장 및 로깅 ===
torch.save(alpha.detach(), "alpha.pt")
alpha_repr = "|".join(map(lambda x: f"{x:.6f}", alpha.detach().cpu().numpy().flatten().tolist()))

print(f"[Selected Pragmas] {selected_factors} | Score: {score:.6f} | Loss: {loss.item():.6f}")
print(f"[Alpha] {alpha_repr}")

# === 기록 ===
append_results_with_metrics(
    pragmas=selected_factors,
    interval=interval,
    results_fields=results_fields,
    power=power,
    score=score,
    alpha_repr=alpha_repr,
    loss=loss.item()
)

update_final_results(
    results_fields=results_fields,
    power=power,
    score=score,
    alpha_repr=alpha_repr,
    loss=loss.item()
)

