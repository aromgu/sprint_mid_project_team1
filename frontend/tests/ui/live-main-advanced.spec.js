import { expect, test } from "@playwright/test";

test.skip(process.env.RUN_LIVE_RAG_E2E !== "true", "RUN_LIVE_RAG_E2E=true에서만 실제 API를 호출합니다.");
test.setTimeout(240_000);

const emptyAnalysis = {
  overview: { submission_deadline: null, inquiry_deadline: null, eligibility_summary: "review_required", risk_counts: {}, deliverable_progress: {}, action_items: [] },
  risks: { risks: [] }, eligibility: { items: [] }, deliverables: { items: [] }, requirements: { items: [] },
};

test("실제 FastAPI와 OpenAI Main Advanced 질의 및 Evaluation 화면", async ({ page }) => {
  // 화면 진입 시 자동 실행되는 탭별 생성 호출은 이 Q&A 통합 테스트의 범위가
  // 아니므로 고정한다. 문서 목록, ask/stream, conversation, evaluation은 실제 API다.
  await page.route("**/api/**", async route => {
    const pathname = new URL(route.request().url()).pathname;
    const key = ["overview", "risks", "eligibility", "deliverables", "requirements"].find(name => pathname.endsWith(`/${name}`));
    if (key) return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(emptyAnalysis[key]) });
    if (pathname === "/api/toc") return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ items: [] }) });
    return route.continue();
  });

  await page.goto("/");
  await expect(page.getByRole("button", { name: "AI 질문", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "AI 질문", exact: true }).click();
  await page.getByRole("combobox", { name: "LLM 선택" }).selectOption("openai");
  await page.getByPlaceholder(/실격 조건과 필수 제출물/).fill("입찰 참가 신청 시 나라장터 업종코드는 무엇인가?");
  const responsePromise = page.waitForResponse(response => response.url().includes("/ask/stream") && response.status() === 200);
  await page.getByRole("button", { name: "질문하기" }).click();
  await responsePromise;
  await expect(page.getByText(/1468|컴퓨터 관련 서비스사업/).first()).toBeVisible({ timeout: 180_000 });

  const resetPromise = page.waitForResponse(response => response.url().includes("/conversation/") && response.status() === 200);
  await page.getByRole("button", { name: "대화 초기화" }).click();
  await resetPromise;
  await expect(page.getByPlaceholder(/실격 조건과 필수 제출물/)).toHaveValue("");

  await page.goto("/internal/evaluation");
  await expect(page.getByRole("heading", { name: "Main Advanced 파이프라인 품질 평가" })).toBeVisible();
  await expect(page.getByText("Main Advanced 평가 완료")).toBeVisible();
  await page.getByRole("button", { name: "시스템 모듈" }).click();
  await expect(page.getByText("Advanced Chroma", { exact: true }).first()).toBeVisible();
});
