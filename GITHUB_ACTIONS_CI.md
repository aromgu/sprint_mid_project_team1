# Git Push마다 CI Pipeline 실행하기

`.github/workflows/ci.yml` 파일을 만들고 아래 내용을 작성한다.

```yaml
name: CI Pipeline

on:
  push:
  pull_request:
    branches:
      - main

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
          python-version: "3.11"

      - name: Install Python dependencies
        run: uv sync --dev --locked

      - name: Run backend tests
        run: uv run pytest --junitxml=pytest-report.xml

      - name: Upload test report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: pytest-report
          path: pytest-report.xml

      - name: Set up Node.js
        uses: actions/setup-node@v4
        with:
          node-version: "22"
          cache: npm
          cache-dependency-path: frontend/package-lock.json

      - name: Install frontend dependencies
        working-directory: frontend
        run: npm ci

      - name: Build frontend
        working-directory: frontend
        run: npm run build
```

## 실행 조건

다음 설정은 모든 브랜치의 `git push`마다 Pipeline을 실행한다.

```yaml
on:
  push:
```

`main`과 `develop` 브랜치의 push에서만 실행하려면 다음처럼 제한한다.

```yaml
on:
  push:
    branches:
      - main
      - develop
```

## 현재 Workflow 참고

현재 `.github/workflows/rag-test.yml`은 저장소에 없는 `requirements.txt`와
`test_rag_pipeline.py`를 참조한다. 위 Workflow는 현재 프로젝트 설정에 맞게
`uv sync --dev --locked`, `uv run pytest`, `npm ci`, `npm run build`를 사용한다.
