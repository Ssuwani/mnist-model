## MNIST Model

```bash
.
├── .github
│   └── workflows
│       ├── cd.yml
│       └── ci.yml
├── dockerfile
└── train.py
```

- `train.py` <br/>
    MNIST Train을 위한 파일입니다.
- `dockerfile` <br/>
    도커 이미지 생성을 위한 설정파일입니다.
- `.github/workflows/ci.yml`<br/>
    지속적 통합을 위한 파일로 레포지토리내 코드를 포매팅합니다.
- `.github/workflows/cd.yml` <br/>
    지속적 배포를 위한 파일로 Experiment Pipeline Workflow 혹은 Deploy Pipeline Workflow를 트리거합니다.

### Train.py

MNIST Train을 위한 파일입니다.

| Input          | Description                                                  |
| -------------- | ------------------------------------------------------------ |
| --hidden_units | (int) 첫번째 히든 레이어의 Units 수입니다.                   |
| --optimizer    | (str) 모델 최적화 함수입니다.                                |
| --save_model   | (str) 모델을 저장시킬 위치를 입력합니다. 미 입력시 모델을 저장하지 않습니다. |

| Output | Description                                        |
| ------ | -------------------------------------------------- |
| model  | Tensorflow의 Saved Model 형태로 모델을 저장합니다. |

| Print                    | Description                                                  |
| ------------------------ | ------------------------------------------------------------ |
| f"test-loss={test-loss}" | Katib Experiment의 Metrics Collection 방식 중 하나인 StdOut의 규칙에 따라 실험 결과에 `print` 함수를 사용합니다. |
| f"test-acc={test-acc}"   | 위와 동일                                                    |

t