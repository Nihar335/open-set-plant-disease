# Voice input for the plant name: interface contract

**Status (2026-10-15): OPERATIONAL LOCALLY, NOT VERIFIED.**
- Local, offline speech-to-text (Vosk) runs in an isolated worker on this computer (`stage10/app/speech/`).
- English and Hindi work end to end in automated tests and in the browser.
- **Not yet verified:** no tests with farmers and no measurement on farmer speech. Hindi has not been reviewed by a fluent speaker.

`stage10/workflow/voice.py` `voice_input_available()` is `True` only when the configured local engines are installed. Otherwise `PlantNameSession.enter(..., channel="voice_transcript_confirmed")` is refused with "voice input is not available".

## Contract (binding)

1. **Transcription.** The engine returns text, an engine identity and a confidence value. The confidence is used only internally, for the low-confidence gate, and is never shown as a percentage.
2. **Confirmation.** The recognised text is shown to the user, who can **use it, edit it, have it read back, say it again or cancel**. Unconfirmed text is never used. An edited transcript is recorded as `typed_text_confirmed`.
3. **Same rules as typed text.** Only confirmed text goes to `PlantNameSession.enter(text, channel="voice_transcript_confirmed")`. It is resolved by exactly the same rules as typing: only verified knowledge-base names, and ambiguous, partial or unknown names need a choice.
4. **English recognition aid.** For English plant names the recogniser is limited to the knowledge base's own names. This can still return a confidently wrong plant name: synthetic "tomato plant" was heard as "grapevine". Confirmation is therefore mandatory, and plant confirmation follows as well.
5. **Privacy.** Audio stays on this computer and in memory only. Nothing is stored, logged or sent anywhere. The worker has networking disabled.
6. **Provenance.** The workflow record keeps `entry_channel = "voice_transcript_confirmed"`.
