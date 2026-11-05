// Turns the structured report (from the server's report engine) into a display model. It never adds findings,
// never shows scores, and keeps the origin of every statement: user-given, photo model, knowledge base.
import { pick } from "./i18n.js";

const TONE = { known_disease: "warn", unknown_investigation: "info", out_of_model_scope: "info", insufficient_image: "bad" };

export function buildReportModel(resp, lang = "en") {
  const r = resp.report;
  const wf = resp.workflow;
  const sym = Object.fromEntries(r.visible_symptoms.map((s) => [s.symptom_id, s]));
  const recs = {};
  for (const g of ["recommended_tests", "treatment_guidance", "nutrition_guidance", "soil_guidance", "immediate_actions"])
    for (const x of r[g]) recs[x.recommendation_id] = x;
  const ran = r.model_prediction.ran;
  const userSymptoms = r.visible_symptoms.filter((s) => s.origin === "user_report");
  const model = {
    reportId: r.report_id,
    type: r.report_type,
    tone: TONE[r.report_type] || "info",
    bannerKey: "report.banner." + r.report_type,
    plant: {
      name: r.plant_identity.crop_confirmed_by_user,
      userGiven: wf.plant_selection.source === "user_supplied",
      verified: false,
      inModelScope: r.plant_identity.in_model_scope,
      entered: wf.plant_selection.entered_text,
    },
    photoCheck: {
      ran,
      reason: wf.routing.reason,
      decision: ran ? r.osr_status.decision : null,
      predictedClass: ran ? r.model_prediction.predicted_class : null,
      agrees: r.plant_identity.model_supporting_evidence.agrees_with_user,
      message: ran ? pick(r.osr_status.message, lang) : null,
    },
    candidates: r.candidate_diagnoses.map((c) => ({
      id: c.candidate_id,
      name: pick(c.candidate_name, lang),
      crop: c.affected_crop,
      fromModel: c.ranking_method === "model_class_evidence",
      uncertainty: pick(c.uncertainty_statement, lang),
      supporting: c.supporting_symptoms.map((id) => sym[id]).filter(Boolean)
        .map((s) => ({ label: pick(s.label, lang), origin: s.origin, status: s.evidence_status })),
      tests: c.recommended_confirmatory_tests.map((id) => recs[id]).filter(Boolean).map((x) => pick(x.text, lang)),
      sources: c.source_ids,
    })),
    symptoms: {
      userSeen: userSymptoms.filter((s) => s.evidence_status === "observed").map((s) => pick(s.label, lang)),
      userAbsent: userSymptoms.filter((s) => s.evidence_status !== "observed").map((s) => pick(s.label, lang)),
      image: r.visible_symptoms.filter((s) => s.origin === "image_analysis").map((s) => pick(s.label, lang)),
      typical: r.visible_symptoms.filter((s) => s.origin === "knowledge_base_typical").map((s) => pick(s.label, lang)),
    },
    actions: r.immediate_actions.map((x) => ({ text: pick(x.text, lang), category: x.category, subcategory: x.subcategory })),
    tests: r.recommended_tests.map((x) => pick(x.text, lang)),
    management: [...r.treatment_guidance, ...r.nutrition_guidance, ...r.soil_guidance].map((x) => pick(x.text, lang)),
    insufficient: r.insufficient_evidence_statement ? pick(r.insufficient_evidence_statement, lang) : null,
    uncertainty: { level: r.uncertainty.level, statements: r.uncertainty.statements.map((s) => pick(s, lang)) },
    warnings: r.safety_warnings.map((w) => pick(w, lang)),
    disclaimer: pick(r.disclaimer, lang),
    sources: r.evidence_sources.map((s) => ({ id: s.source_id, title: s.title, by: s.publisher_or_author, url: s.url_or_doi })),
    expertReferral: r.report_type !== "known_disease" || !r.candidate_diagnoses.length ||
      r.immediate_actions.some((x) => x.subcategory === "consult_expert"),
    expertReviewed: r.review_status.expert_review === "reviewed",
    hindiDraft: r.language === "hi",
    retake: (resp.image_quality && resp.image_quality.retake_requests) || [],
    highlighting: { available: false },
    marks: ((resp.case && resp.case.stages.user_marked_regions) || { regions: [] }).regions
      .map((r) => ({ ...r.normalized, shape: r.shape, source: r.source, verified: r.verified })),
  };
  return model;
}
