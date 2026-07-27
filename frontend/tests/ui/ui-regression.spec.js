import { expect, test } from "@playwright/test";

const evidence = { document_name: "테스트 RFP", page_number: 7, quote: "카드 근거", score: 0.92, requirement_ids: ["R-001"] };
const responses = {
  documents: [{ document_id: "ui-001", title: "UI 회귀 테스트 RFP", organization: "테스트 기관", difficulty: "high", document_date: "20260727", status: "ready" }],
  toc: { document_id: "ui-001", items: [{ id: "toc-1", title: "입찰 참가자격", page: 7 }] },
  overview: { document_id: "ui-001", submission_deadline: "2026.08.10. 16:00", inquiry_deadline: "2026.08.05. 18:00", eligibility_summary: "확인 필요", confidence: 1, action_items: [] },
  risks: { document_id: "ui-001", risks: [{ id: "risk-1", type: "disqualification", severity: "critical", title: "필수 서류 누락", description: "미제출 시 입찰 무효", user_status: "unchecked", evidence }] },
  eligibility: { document_id: "ui-001", items: [{ id: "el-1", title: "참가 자격", description: "등록 자격 확인", user_status: "unchecked", evidence }] },
  deliverables: { document_id: "ui-001", items: [{ id: "del-1", name: "제안서", kind: "bid_submission", description: "전자 제출", status: "pending", quantity: 1, evidence }, { id: "del-2", name: "최종보고서", kind: "project_deliverable", description: "검수 전 납품", status: "pending", quantity: 1, evidence }] },
  requirements: { document_id: "ui-001", items: [{ id: "SFR-001", category: "functional", title: "검색 기능", description: "문서 검색 제공", priority: "high", review_status: "pending", evidence }] },
};

test.beforeEach(async ({ page }) => {
  await page.route("**/api/**", async route => {
    const url = new URL(route.request().url()); let body = {};
    if (url.pathname === "/api/documents") body = responses.documents;
    else if (url.pathname === "/api/toc") body = responses.toc;
    else if (url.pathname.endsWith("/overview")) body = responses.overview;
    else if (url.pathname.endsWith("/risks")) body = responses.risks;
    else if (url.pathname.endsWith("/eligibility")) body = responses.eligibility;
    else if (url.pathname.endsWith("/deliverables")) body = responses.deliverables;
    else if (url.pathname.endsWith("/requirements")) body = responses.requirements;
    else if (url.pathname === "/api/document") body = { document_id: "ui-001", page: 7, page_count: 20, text: "입찰 참가자격과 제출 조건이 기재된 실제 원문 영역", headings: [] };
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
  });
});

test("Overview 요약카드가 업무 탭으로 이동한다", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("탭별 핵심 요약")).toBeVisible();
  await page.getByRole("button", { name: /GO \/ NO-GO/ }).click();
  await expect(page.getByText("참가 자격", { exact: true }).first()).toBeVisible();
  await page.getByRole("button", { name: "Overview" }).click();
  await page.getByRole("button", { name: /실행 준비 제출물/ }).click();
  await expect(page.getByText("입찰 제출서류")).toBeVisible();
  await expect(page.getByText("사업 수행 산출물")).toBeVisible();
  await page.getByRole("button", { name: "Overview" }).click();
  await page.getByRole("button", { name: /요구사항 전체/ }).click();
  await expect(page.getByText("이 문서가 요구하는 것")).toBeVisible();
  await page.getByRole("button", { name: "Overview" }).click();
  await page.getByRole("button", { name: /AI 질문 문서 근거/ }).click();
  await expect(page.getByPlaceholder(/실격 조건과 필수 제출물/)).toBeVisible();
});

test("파일 목록을 접고 이름 검색과 날짜·이름 정렬만 제공한다", async ({ page }) => {
  await page.goto("/");
  const toggle = page.getByRole("button", { name: "파일 목록 열기" });
  await expect(toggle).toBeVisible();
  await toggle.click();
  await expect(page.getByPlaceholder("RFP 검색")).toBeVisible();
  const sort = page.getByRole("combobox", { name: "파일 정렬" });
  await expect(sort.locator("option")).toHaveText(["날짜순", "이름순"]);
  await expect(page.getByText("전체 난이도")).toBeHidden();
});

test("문서 목차에서 원문 페이지를 연다", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "문서 목차" }).click();
  await expect(page.getByText("1개")).toBeVisible();
  await page.getByRole("button", { name: /입찰 참가자격/ }).click();
  await expect(page.getByText("원문 페이지")).toBeVisible();
  await expect(page.getByText(/실제 원문 영역/)).toBeVisible();
});

test("Evidence 폭을 드래그하고 새로고침 후 유지한다", async ({ page }) => {
  await page.goto("/");
  const evidencePanel = page.locator(".evidence"); const separator = page.getByRole("separator");
  const before = await evidencePanel.boundingBox(); const handle = await separator.boundingBox();
  await page.mouse.move(handle.x + handle.width / 2, handle.y + 200);
  await page.mouse.down(); await page.mouse.move(handle.x - 120, handle.y + 200); await page.mouse.up();
  const after = await evidencePanel.boundingBox();
  expect(after.width).toBeGreaterThan(before.width + 90);
  await page.reload(); await expect(page.getByText("탭별 핵심 요약")).toBeVisible();
  const restored = await evidencePanel.boundingBox();
  expect(Math.abs(restored.width - after.width)).toBeLessThan(3);
});

test("자격과 제출물 상태를 화면에서 변경한다", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Go / No-Go" }).click();
  const eligibilityStatus = page.locator(".goNoGoView .card select").first();
  await eligibilityStatus.selectOption("met"); await expect(eligibilityStatus).toHaveValue("met");
  await page.getByRole("button", { name: "실행 준비" }).click();
  const deliverableStatus = page.locator(".deliverablesView .card select").first();
  await deliverableStatus.selectOption("completed"); await expect(deliverableStatus).toHaveValue("completed");
  await expect(page.getByText("1개", { exact: true }).nth(1)).toBeVisible();
});

test("입찰 제출서류와 사업 수행 산출물을 개별 구역에 표시한다", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "실행 준비" }).click();
  await expect(page.getByText("입찰 제출서류")).toBeVisible();
  await expect(page.getByText("제안서", { exact: true })).toBeVisible();
  await expect(page.getByText("사업 수행 산출물")).toBeVisible();
  await expect(page.getByText("최종보고서", { exact: true })).toBeVisible();
});

test("좁은 화면에서는 Evidence를 숨기고 내용이 넘치지 않는다", async ({ page }) => {
  await page.setViewportSize({ width: 980, height: 900 }); await page.goto("/");
  await expect(page.locator(".evidence")).toBeHidden(); await expect(page.locator(".evidenceResize")).toBeHidden();
  const overflow = await page.locator(".layout").evaluate(node => node.scrollWidth > node.clientWidth);
  expect(overflow).toBe(false);
});

test("Overview 데스크톱 스크린숏", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("탭별 핵심 요약")).toBeVisible();
  await expect(page).toHaveScreenshot("overview-desktop.png", { fullPage: true, animations: "disabled" });
});

test("Go-NoGo와 요구사항 화면 스크린숏", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Go / No-Go" }).click();
  await expect(page).toHaveScreenshot("go-no-go-desktop.png", { fullPage: true, animations: "disabled" });
  await page.getByRole("button", { name: "요구사항" }).click();
  await expect(page).toHaveScreenshot("requirements-desktop.png", { fullPage: true, animations: "disabled" });
});
