📘 Week 1 상세 계획
✅ Day 1–2: 실험 구조 세분화 및 Loss Function 설계
기존 MMD + CE 구조와 구별되는 변분 정렬 loss function을 명확히 구현

실험 플래그: --source_mmd, --pseudo_prior, --kl_reg, --pe_sampling

--------------------------
✅ Day 3–4: Prompt Generator 구조 설계
GPF/MLP 대신 prior generator $q_\phi$ 구조 설계

PE-anchor 기반

GMM 또는 encoder-space projection 기반

구현 완료 후 prompt_generator.py로 분리
--------------------------


✅ Day 5–6: Baseline & Loader 구축
CE-only, MMD-to-source (oracle), Ours (target-inductive prior)

Pretrained encoder 로딩 + frozen wrapper

실험 config 템플릿 작성

✅ Day 7: Validation loss 시각화 코드
Validation accuracy, CE loss, KL/MMD loss trend를 모두 그래프로 plot

논문화 대비 시각화 코드 미리 준비

📗 Week 2 상세 계획
✅ Day 8–10: 핵심 실험 실행
(a) GPF baseline

(b) MLP prompt baseline

(c) Ours (pseudo-prior generator + CE + KL)

Target = Cora, Citeseer, Pubmed / 1-shot, 5-shot 분리

✅ Day 11–12: 결과 비교 + Ablation 실험
Without KL, without CE, different prior shapes 비교

각 구성 요소가 성능에 미치는 영향 분석

✅ Day 13: 실험표/그래프 정리
정량표 (accuracy, MMD, variance)

실험 그룹별 trend 요약

✅ Day 14: 논문 구도 재정리 + 지도교수 보고
"이 방향으로 충분히 novel하고 정량적 근거 있음"이라는 message 강조

SIGL/ICLR 워크샵 short-paper 제출 고려

🔩 보조 체크리스트 (ISTP-style 엔지니어링 관점)
 실험 config 관리 yaml화

 pretrained encoder의 geometric fingerprint log 저장

 prior generator의 분포 시각화 (UMAP/TSNE)

 코드: trainer.py, losses.py, prior.py, prompt.py 구조 정리

