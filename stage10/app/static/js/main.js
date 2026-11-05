import { boot } from "./app.js";

boot().then(async (ok) => {
  // In-browser UI self-test (development only): open /?selftest=1#/home
  if (ok && new URLSearchParams(location.search).get("selftest") === "1") {
    const { runSelfTest } = await import("./selftest.js");
    runSelfTest();
  }
});
