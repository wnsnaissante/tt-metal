# Mamba SSD Recurrence Fused Op 계획

## 목표

`ssd_forward()` 안의 inter-chunk recurrence 블록을 `/home/wormhole/tt-metal` 쪽 TT-NN fused/custom op로 치환해서

- dispatch 수를 줄이고
- `reshape / permute / slice` 같은 shape-op 오버헤드를 줄이고
- Python 쪽에서 흩어진 recurrence 계산을 하나의 전용 op로 묶는 것

을 목표로 한다.

적용 범위는 다음과 같다.

- Python 측:
  - [mamba2.py](/home/wormhole/ttnn-netmamba2/src/modules/mamba2.py)
  - `ssd_forward()` 안의 recurrence 블록

- TT-Metal 측:
  - `/home/wormhole/tt-metal/ttnn/cpp/ttnn/operations/experimental`

## 현재 Python 입력 계약

현재 recurrence 입력은 아래 텐서들 기준으로 정리되어 있다.

- `states_bhcpn`
  - shape: `[B, H, C, P, N]`
  - 의미:
    - `B`: batch
    - `H`: num_heads
    - `C`: num_chunks
    - `P`: head_dim
    - `N`: state_size

- `initial_states`
  - shape: `[B, H, 1, P, N]`
  - recurrence용 canonical shape로 맞춰진 초기 state

- `A_end_bhc`
  - shape: `[B, H, C]`
  - 각 chunk 끝에서의 누적 감쇠값

즉 fused op는 이 세 텐서를 직접 입력으로 받는 형태로 설계하는 것이 가장 자연스럽다.

## 권장 fused op 범위

`ssd_forward()` 전체를 한 번에 fused 하지 않는다.

1차 fused op는 아래 recurrence 부분만 담당하는 것이 맞다.

- recurrence decay 생성/적용
- chunk 간 state 전파
- 결과 반환:
  - chunk별 갱신된 states
  - final_state

권장 op 이름:

- `ttnn.experimental.mamba_ssd_recurrence`

권장 입력:

- `states_bhcpn: [B, H, C, P, N]`
- `initial_state_bh1pn: [B, H, 1, P, N]`
- `a_end_bhc: [B, H, C]`

권장 출력:

- `states_bhcpn_out: [B, H, C, P, N]`
- `final_state_bhpn: [B, H, P, N]`

즉 1차 버전에서는 `Y_diag`와 `Y_off`는 Python 쪽에 그대로 두고, recurrence만 분리한다.

## 왜 이 범위가 맞는가

현재 SSD 안에서 남은 최적화 후보 중, 이 recurrence 블록이 제일 큰 구조/dispatch 병목 후보다.

이 블록에는 아직 아래 연산들이 연속해서 남아 있다.

- `concat`
- `segsum`
- `exp`
- `reshape`
- `matmul`
- `slice`

즉 아래 후보들보다 ROI가 더 좋다.

- `split_mamba_proj` fused op
- `split_xbc` fused op
- `ssd_forward` 전체 fused op

## TT-Metal 안에서 둘 위치

권장 위치는 `experimental` 아래의 `mamba` 카테고리다.

### API 파일

- `/home/wormhole/tt-metal/ttnn/cpp/ttnn/operations/experimental/mamba/mamba_ssd_recurrence.hpp`
- `/home/wormhole/tt-metal/ttnn/cpp/ttnn/operations/experimental/mamba/mamba_ssd_recurrence.cpp`

### Device op 파일

- `/home/wormhole/tt-metal/ttnn/cpp/ttnn/operations/experimental/mamba/device/mamba_ssd_recurrence_device_operation.hpp`
- `/home/wormhole/tt-metal/ttnn/cpp/ttnn/operations/experimental/mamba/device/mamba_ssd_recurrence_device_operation.cpp`
- `/home/wormhole/tt-metal/ttnn/cpp/ttnn/operations/experimental/mamba/device/mamba_ssd_recurrence_device_operation_types.hpp`
- `/home/wormhole/tt-metal/ttnn/cpp/ttnn/operations/experimental/mamba/device/mamba_ssd_recurrence_program_factory.hpp`
- `/home/wormhole/tt-metal/ttnn/cpp/ttnn/operations/experimental/mamba/device/mamba_ssd_recurrence_program_factory.cpp`

### Nanobind 파일

- `/home/wormhole/tt-metal/ttnn/cpp/ttnn/operations/experimental/mamba/mamba_ssd_recurrence_nanobind.hpp`
- `/home/wormhole/tt-metal/ttnn/cpp/ttnn/operations/experimental/mamba/mamba_ssd_recurrence_nanobind.cpp`

## 참고할 패턴

구조를 참고할 때는 아래 파일들이 가장 적절하다.

### 단순 API / composite 패턴

- [example.hpp](/home/wormhole/tt-metal/ttnn/cpp/ttnn/operations/examples/example/example.hpp)
- [example.cpp](/home/wormhole/tt-metal/ttnn/cpp/ttnn/operations/examples/example/example.cpp)

### stateful / device op 패턴

- [kv_cache.hpp](/home/wormhole/tt-metal/ttnn/cpp/ttnn/operations/kv_cache/kv_cache.hpp)
- [kv_cache.cpp](/home/wormhole/tt-metal/ttnn/cpp/ttnn/operations/kv_cache/kv_cache.cpp)
- [update_cache_device_operation.cpp](/home/wormhole/tt-metal/ttnn/cpp/ttnn/operations/kv_cache/device/update_cache_device_operation.cpp)

### Nanobind 패턴

- [example_nanobind.cpp](/home/wormhole/tt-metal/ttnn/cpp/ttnn/operations/examples/example/example_nanobind.cpp)

### CMake 패턴

- [examples/CMakeLists.txt](/home/wormhole/tt-metal/ttnn/cpp/ttnn/operations/examples/CMakeLists.txt)

## 구현 순서

1. `mamba_ssd_recurrence`의 C++ API 추가
2. device operation types / validate 추가
3. output spec / output tensor 생성 로직 추가
4. program factory 추가
5. nanobind binding 추가
6. CMake 연결
7. TT-Metal 빌드/설치
8. Python의 `ssd_forward()` recurrence 블록만 새 op 호출로 교체

## Python 쪽 교체 계획

[mamba2.py](/home/wormhole/ttnn-netmamba2/src/modules/mamba2.py) 에서 현재 아래 블록을:

- `states_with_init_bhcpn`
- `A_end_bhc`
- `decay_chunk`
- `states_with_init_flat`
- `decay_chunk_flat`
- `new_states`
- `final_state`
- `states`

아래 같은 한 호출로 교체하는 것을 목표로 한다.

```python
states_bhcpn, final_state = ttnn.experimental.mamba_ssd_recurrence(
    states_bhcpn,
    initial_states,
    A_end_bhc,
)
```

이후 Python에서는 반환된 `states_bhcpn`을 `Y_off` 경로에 맞게만 소비한다.

## 검증 계획

### 1. 정확도 검증

현재 Python recurrence 출력과 fused-op 출력을 비교한다.

비교 대상:

- `states`
- `final_state`

처음에는 작은 shape와 고정 seed 기준으로 확인한다.

### 2. shape 계약 검증

입출력 shape가 아래와 정확히 일치하는지 확인한다.

- `states_bhcpn`
- `initial_state_bh1pn`
- `a_end_bhc`

### 3. 성능 검증

아래를 다시 측정한다.

- [test_mamba2_throughput.py](/home/wormhole/ttnn-netmamba2/src/tests/test_mamba2_throughput.py)
- SSD 전용 trace 경로

그리고 아래를 비교한다.

- dispatch 수
- end-to-end throughput

## 1차 버전에서 하지 않을 것

v1에서는 아래를 포함하지 않는다.

- `Y_off` fusion
- `Y_diag` fusion
- `split_mamba_proj` fusion
- `split_xbc` fusion
- `ssd_forward` 전체 fusion

## 기대 효과

1차 버전에서 기대하는 최선의 효과는 아래와 같다.

- recurrence dispatch 감소
- Python 쪽 `reshape / permute / slice` 감소
- 다음 단계인 `Y_off` fused op로 가기 위한 경로 정리

이게 성공하면, 다음 fused target으로 가장 유력한 건:

- `states -> Y_off`

이다.
