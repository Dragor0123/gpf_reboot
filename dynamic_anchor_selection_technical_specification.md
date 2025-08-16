# Dynamic Performance-Based Anchor Selection: Technical Specification

## 0. 개요 (Overview)

Dynamic Performance-Based Anchor Selection은 그래프 신경망의 크로스 도메인 전이 학습에서 사용되는 새로운 앵커 선택 방법입니다. 기존의 정적 앵커 방법들과 달리, 이 시스템은 **현재 모델의 성능을 기반으로 동적으로 높은 성능을 보이는 노드들을 앵커로 선택**합니다.

### 핵심 특징:
- **성능 기반 선택**: 개별 노드의 손실값과 예측 신뢰도를 바탕으로 앵커 선정
- **동적 적응**: 훈련 진행에 따라 앵커가 동적으로 업데이트됨
- **소스 프리 운영**: 소스 도메인 데이터에 의존하지 않음
- **Fallback 메커니즘**: 동적 선택 실패 시 안전한 대안 제공

### 기존 방법 대비 혁신성:
- **MoG (Mixture of Gaussians)**: 정적으로 생성된 가우시안 분포 기반 앵커
- **Random Selection**: 무작위 노드 선택
- **Dynamic Performance**: **실제 태스크 성능을 반영한 지능적 선택**

---

## 1. Input/Output 명세

### Input
```python
def select_anchors_by_performance(
    embeddings: torch.Tensor,    # [N, D] - 모든 노드의 임베딩
    logits: torch.Tensor,        # [N, C] - 모델 예측 결과
    labels: torch.Tensor,        # [N] - 실제 레이블
    mask: torch.Tensor,          # [N] - 훈련 노드 마스크 (bool)
    epoch: int                   # 현재 에포크 번호
) -> Tuple[torch.Tensor, Dict[str, Any]]
```

### Output
```python
return (
    anchor_embeddings,  # torch.Tensor [K, D] - 선택된 앵커 임베딩들
    selection_info      # Dict - 선택 과정의 메타데이터 및 통계
)
```

### selection_info 구조:
```python
{
    'epoch': int,                    # 선택이 수행된 에포크
    'num_anchors': int,              # 선택된 앵커 개수
    'selection_ratio': float,        # 사용된 선택 비율
    'selected_indices': List[int],   # 선택된 노드들의 전역 인덱스
    'anchor_shape': List[int],       # 앵커 텐서의 형태
    'update_type': str,              # 'soft' 또는 'hard' 업데이트
    'quality_metrics': Dict,         # 앵커 품질 지표들
    'performance_stats': Dict        # 성능 통계
}
```

---

## 2. config.yaml의 dynamic_anchor 섹션 Hyperparameters 상세 설명

### 2.1 기본 설정 (Basic Configuration)

```yaml
dynamic_anchor:
  enable: true                      # 동적 앵커 선택 활성화
  selection_ratio: 0.3              # 앵커로 선택할 노드 비율 (30%)
  update_frequency: 10              # 앵커 업데이트 주기 (10 에포크마다)
  soft_update_momentum: 0.9         # 소프트 업데이트 모멘텀
```

#### 파라미터 설명:

**`enable` (bool, default: true)**
- 동적 앵커 선택 기능의 활성화 여부
- `false`로 설정 시 레거시 정적 방법 사용

**`selection_ratio` (float, range: (0, 1], default: 0.3)**
- 전체 훈련 노드 중 앵커로 선택할 비율
- 예: 0.3 = 상위 30% 성능 노드 선택
- 값이 클수록 더 많은 앵커, 작을수록 더 선별적 선택
- **영향**: 0.1-0.2 (보수적), 0.3-0.5 (표준), 0.6+ (공격적)

**`update_frequency` (int, default: 10)**
- 앵커 업데이트 간격 (에포크 단위)
- 더 작은 값 = 더 자주 업데이트 (더 동적)
- 더 큰 값 = 덜 자주 업데이트 (더 안정적)
- **권장 범위**: 5-20 에포크

**`soft_update_momentum` (float, range: [0, 1], default: 0.9)**
- 앵커 업데이트 시 이전 값과의 혼합 비율
- 0.0 = 완전 교체 (hard update)
- 1.0 = 변경 없음
- 0.9 = 이전 90% + 새로운 10%
- **훈련 안정성에 중요한 역할**

### 2.2 성능 평가 기준 (Performance Criteria)

```yaml
criteria:
  loss_weight: 1.0                  # 개별 노드 손실 가중치
  confidence_weight: 0.0            # 예측 신뢰도 가중치
```

**`loss_weight` (float, default: 1.0)**
- 개별 노드 손실값 기반 성능 점수의 가중치
- 낮은 손실 = 높은 성능 점수
- **주요 선택 기준**: 실제 태스크 성능을 직접 반영

**`confidence_weight` (float, default: 0.0)**
- 예측 신뢰도(낮은 엔트로피) 기반 성능 점수의 가중치
- 높은 신뢰도 = 안정적인 예측
- **현재 설정**: 손실값만 사용 (confidence_weight=0.0)

**제약 조건**: `loss_weight + confidence_weight = 1.0`

### 2.3 품질 추적 (Quality Tracking)

```yaml
quality_tracking: true              # 앵커 품질 메트릭 추적 활성화
```

**`quality_tracking` (bool, default: true)**
- 앵커 선택의 품질을 모니터링하는 기능
- 활성화 시 다음 메트릭들을 계산:
  - **Diversity**: 선택된 앵커들 간의 다양성
  - **Representativeness**: 전체 분포 대표성
  - **Stability**: 연속 선택 간 일관성
  - **Coverage**: 앵커들의 데이터 커버리지

---

## 3. Fallback 로직 상세 설명

### 3.1 Fallback 설정

```yaml
fallback:
  enable: true                      # Fallback 메커니즘 활성화
  method: "mog"                     # Fallback 방법 (MoG 앵커 사용)
  conditions:                       # Fallback 트리거 조건들
    performance_drop: 0.05          # 성능 하락 임계값 (5%)
    anchor_diversity: 0.3           # 최소 앵커 다양성 임계값
    selection_instability: 0.5      # 최소 선택 안정성 임계값 (50% 오버랩)
```

### 3.2 Fallback 트리거 조건

#### 조건 1: 성능 하락 감지 (Performance Drop)
```python
if self.last_performance is not None:
    performance_drop = self.last_performance - current_performance
    if performance_drop > performance_drop_threshold:  # 0.05
        trigger_fallback("Performance dropped by {:.3f}".format(performance_drop))
```

**동작 원리**:
- 이전 에포크 대비 정확도가 5% 이상 하락 시 트리거
- 예: 95% → 89% 정확도 변화 시 fallback 활성화

#### 조건 2: 앵커 다양성 부족 (Low Anchor Diversity)
```python
diversity_score = compute_pairwise_distances(anchors).mean()
if diversity_score < anchor_diversity_threshold:  # 0.3
    trigger_fallback("Anchor diversity too low: {:.3f}".format(diversity_score))
```

**동작 원리**:
- 선택된 앵커들이 너무 유사한 경우 (클러스터링)
- 평균 쌍별 거리가 0.3 미만 시 트리거

#### 조건 3: 선택 불안정성 (Selection Instability)
```python
overlap_ratio = compute_anchor_overlap(current_anchors, previous_anchors)
if overlap_ratio < selection_instability_threshold:  # 0.5
    trigger_fallback("Selection too unstable: {:.3f} overlap".format(overlap_ratio))
```

**동작 원리**:
- 연속된 앵커 선택 간 오버랩이 50% 미만 시 트리거
- 앵커 선택이 너무 변동적일 때 안정성 확보

### 3.3 Fallback 활성화 과정

1. **조건 검사**: 매 앵커 업데이트 시점에 3가지 조건 검사
2. **Fallback 활성화**: 조건 위반 시 즉시 MoG 앵커로 전환
3. **상태 추적**: `is_using_fallback=True`, `fallback_reason` 기록
4. **로그 출력**: 상세한 fallback 활성화 이유 기록

### 3.4 Fallback 해제 과정

```python
# 다음 업데이트 주기에서 조건 재검사
if not should_fallback and self.is_using_fallback:
    self._deactivate_fallback()
    # 동적 앵커 선택으로 복귀
```

**자동 복구**: 조건이 개선되면 자동으로 동적 선택으로 복귀

---

## 4. Dynamic Anchor 선정 기준 및 프로세스

### 4.1 노드 성능 평가 프로세스

#### Step 1: 개별 노드 손실 계산
```python
def compute_loss_scores(logits, labels, mask):
    # 개별 노드별 cross-entropy 손실 계산
    node_losses = F.cross_entropy(logits, labels, reduction='none')  # [N]
    train_losses = node_losses[mask]  # 훈련 노드만 선택
    
    # 정규화: 낮은 손실 → 높은 점수
    min_loss, max_loss = train_losses.min(), train_losses.max()
    normalized = (train_losses - min_loss) / (max_loss - min_loss + 1e-8)
    scores = 1.0 - normalized  # 역순: 낮은 손실 = 높은 점수
    
    return scores
```

#### Step 2: 예측 신뢰도 계산 (현재 비활성화)
```python
def compute_confidence_scores(logits, mask):
    # 예측 확률 계산
    probs = F.softmax(logits, dim=1)
    
    # 엔트로피 계산 (낮은 엔트로피 = 높은 신뢰도)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    confidence = -entropy  # 높은 신뢰도 = 높은 점수
    
    return normalize(confidence[mask])
```

#### Step 3: 점수 결합
```python
combined_scores = (loss_weight * loss_scores + 
                  confidence_weight * confidence_scores)
# 현재 설정: loss_weight=1.0, confidence_weight=0.0
# 따라서 손실값만으로 평가
```

### 4.2 Top-K 선택 프로세스

#### Step 4: 상위 성능 노드 선택
```python
def get_top_performing_indices(logits, labels, mask, selection_ratio):
    # 성능 점수 계산
    scores = compute_combined_scores(logits, labels, mask)
    
    # 선택할 앵커 개수 결정
    num_train_nodes = mask.sum().item()
    num_selected = max(1, int(num_train_nodes * selection_ratio))
    
    # Top-K 선택
    top_scores, top_local_indices = torch.topk(scores, num_selected, largest=True)
    
    # 전역 인덱스로 변환
    masked_node_indices = torch.where(mask)[0]
    top_global_indices = masked_node_indices[top_local_indices]
    
    return top_global_indices
```

### 4.3 구체적인 선정 개수 계산

#### 현재 설정 기준 계산:
- **selection_ratio = 0.3** (30%)
- **훈련 노드 수에 따른 앵커 개수**:

| 훈련 노드 수 | 선택되는 앵커 개수 | 비율 |
|-------------|------------------|------|
| 100개       | 30개             | 30%  |
| 500개       | 150개            | 30%  |
| 1000개      | 300개            | 30%  |
| 2000개      | 600개            | 30%  |

#### 최소/최대 제한:
```python
num_selected = max(1, int(num_train_nodes * selection_ratio))  # 최소 1개
num_selected = min(num_selected, num_train_nodes)              # 최대 전체 노드 수
```

### 4.4 앵커 업데이트 주기별 프로세스

#### 업데이트 주기 판단 (매 에포크마다):
```python
def should_update_anchors(epoch):
    if last_update_epoch == -1:  # 첫 번째 업데이트
        return True
    
    return (epoch - last_update_epoch) >= update_frequency  # 10 에포크마다
```

#### 소프트 업데이트 적용:
```python
if soft_update_momentum > 0 and previous_anchors exists:
    updated_anchors = (0.9 * previous_anchors + 0.1 * new_anchors)
else:
    updated_anchors = new_anchors  # Hard update
```

### 4.5 품질 메트릭 계산

#### 다양성 (Diversity):
```python
# 앵커들 간 평균 쌍별 거리
anchor_distances = torch.cdist(anchors, anchors, p=2)
diversity = anchor_distances[upper_triangle].mean()
```

#### 대표성 (Representativeness):
```python
# 모든 훈련 노드에서 가장 가까운 앵커까지의 평균 거리
distances_to_anchors = torch.cdist(train_embeddings, anchors, p=2)
min_distances = distances_to_anchors.min(dim=1)[0]
representativeness = min_distances.mean()
```

#### 안정성 (Stability):
```python
# 이전 선택과의 오버랩 비율
distances = torch.cdist(current_anchors, previous_anchors, p=2)
threshold = distances.median() * 0.5
matching_anchors = (distances.min(dim=1)[0] <= threshold).float()
stability = matching_anchors.mean()
```

### 4.6 전체 프로세스 타임라인

```
Epoch 0:   [첫 번째 앵커 선택] → 상위 30% 노드 선택
Epoch 1-9: [앵커 유지] → 기존 앵커 사용
Epoch 10:  [앵커 업데이트] → 새로운 상위 30% 선택 + 소프트 업데이트
Epoch 11-19: [앵커 유지]
Epoch 20:  [앵커 업데이트] → 반복...
```

각 업데이트 시점에서:
1. 현재 성능으로 노드들 평가
2. 상위 30% 성능 노드 식별
3. Fallback 조건 검사
4. 소프트 업데이트 적용 (momentum=0.9)
5. 품질 메트릭 계산 및 로깅

이러한 프로세스를 통해 동적으로 가장 높은 성능을 보이는 노드들을 앵커로 선택하여, 모델의 학습 진행에 따라 적응적으로 정규화 기준점을 업데이트합니다.