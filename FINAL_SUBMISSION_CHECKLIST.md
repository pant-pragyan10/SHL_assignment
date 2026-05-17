# Final Submission Checklist

- [ ] API status: `/health` responds OK, `/chat` returns schema-compliant responses
- [ ] Deployment readiness: Dockerfile, entrypoint, env-var validation present
- [ ] Known limitations documented
- [ ] Evaluation summary included (`evaluation/e2e_summary.json` and `evaluation/retrieval_calibration.json`)
- [ ] Production retrieval config documented
- [ ] Behavioral probes pass (see `evaluation/e2e_summary.json`)
- [ ] Hallucination rate negligible on sample queries
- [ ] README and approach doc included
