// Translation interface. All user-facing interface text comes from catalogues (stage10/app/i18n/<lang>.json).
// Missing strings fall back to English; a catalogue that is not "source" or "reviewed" is never presented as verified.

export function createI18n(catalogues, lang = "en") {
  const en = (catalogues.en && catalogues.en.strings) || {};
  const cur = catalogues[lang] || { meta: { review_status: "not_available" }, strings: {} };
  const missing = new Set();

  function format(s, params) {
    return s.replace(/\{(\w+)\}/g, (m, k) => (params && k in params ? String(params[k]) : m));
  }

  function t(key, params) {
    if (cur.strings && Object.prototype.hasOwnProperty.call(cur.strings, key)) return format(cur.strings[key], params);
    if (Object.prototype.hasOwnProperty.call(en, key)) {
      if (lang !== "en") missing.add(key);
      return format(en[key], params);
    }
    missing.add(key);
    return key; // visible, never silently empty
  }

  function status() {
    const rs = (cur.meta && cur.meta.review_status) || "not_available";
    return { lang, review_status: rs, verified: rs === "source" || rs === "reviewed", missing: [...missing] };
  }

  return { t, status, lang };
}

// Localized text objects from the report engine: {en: "...", hi: "..."}.
export function pick(obj, lang = "en") {
  if (obj == null) return "";
  if (typeof obj === "string") return obj;
  return obj[lang] || obj.en || "";
}
