### MNIST Model

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

