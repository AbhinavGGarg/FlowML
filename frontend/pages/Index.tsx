import { useState } from "react";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import WaterLogo from "@/components/WaterLogo";

type HomeTab = "impact" | "problem" | "inspiration";

const Index = () => {
  const [activeTab, setActiveTab] = useState<HomeTab>("impact");

  return (
    <div className="relative min-h-screen overflow-hidden bg-background text-foreground">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(75%_60%_at_15%_5%,rgba(59,130,246,0.30),transparent_65%),radial-gradient(65%_55%_at_85%_12%,rgba(14,116,255,0.22),transparent_65%),linear-gradient(180deg,rgba(2,6,23,1)_0%,rgba(2,8,30,1)_55%,rgba(2,12,42,1)_100%)]" />
      <div className="pointer-events-none absolute -top-20 left-0 h-80 w-full animate-[float_13s_ease-in-out_infinite] bg-[radial-gradient(45%_40%_at_50%_45%,rgba(147,197,253,0.16),transparent_72%)]" />
      <div className="pointer-events-none absolute bottom-0 left-0 h-72 w-full animate-[float_16s_ease-in-out_infinite_reverse] bg-[radial-gradient(50%_55%_at_50%_55%,rgba(37,99,235,0.22),transparent_72%)]" />

      <main className="relative z-10 mx-auto flex min-h-screen w-full max-w-6xl flex-col px-5 pb-12 pt-6 md:px-8">
        <header className="mb-12 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <WaterLogo className="h-11 w-11" />
            <div>
              <p className="text-xl font-semibold tracking-tight text-blue-100">FlowML</p>
            </div>
          </div>
          <Link
            to="/lab"
            className="rounded-full border border-blue-400/45 bg-blue-500/15 px-4 py-2 text-sm font-semibold text-blue-100 transition hover:border-blue-300 hover:bg-blue-500/30"
          >
            Open ML Lab
          </Link>
        </header>

        <motion.section
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55 }}
          className="mb-10"
        >
          <p className="mb-3 text-xs uppercase tracking-[0.26em] text-blue-300/75">FlowML Platform</p>
          <h1 className="max-w-3xl text-4xl font-semibold leading-tight text-blue-50 md:text-6xl">
            Turn raw business data into explainable ML decisions.
          </h1>
          <p className="mt-5 max-w-2xl text-base leading-relaxed text-blue-100/80 md:text-lg">
            FlowML ingests tabular data, runs model selection and training, and presents clear insights your team can trust.
            The experience is designed for speed, clarity, and production-ready outcomes.
          </p>
        </motion.section>

        <motion.section
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1, duration: 0.55 }}
          className="mb-10 grid gap-4 md:grid-cols-3"
        >
          {[
            {
              title: "Ingest",
              copy: "Upload CSV data and define prediction targets in seconds.",
            },
            {
              title: "Train + Evaluate",
              copy: "Run guided model workflows with transparent metrics and stage-by-stage traceability.",
            },
            {
              title: "Deploy Ready",
              copy: "Generate practical artifacts and reports to move from prototype to shipping faster.",
            },
          ].map((item) => (
            <article
              key={item.title}
              className="rounded-2xl border border-blue-500/25 bg-slate-950/55 p-5 shadow-[0_16px_50px_rgba(15,23,42,0.45)] backdrop-blur-sm"
            >
              <h2 className="mb-2 text-lg font-semibold text-blue-100">{item.title}</h2>
              <p className="text-sm leading-relaxed text-blue-100/75">{item.copy}</p>
            </article>
          ))}
        </motion.section>

        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.55 }}
          className="rounded-2xl border border-blue-500/30 bg-slate-950/65 p-5 md:p-7"
        >
          <div className="mb-6 flex flex-wrap gap-3">
            <button
              type="button"
              onClick={() => setActiveTab("impact")}
              className={`rounded-full px-5 py-2 text-sm font-semibold transition ${
                activeTab === "impact"
                  ? "bg-blue-500 text-white shadow-[0_10px_24px_rgba(37,99,235,0.35)]"
                  : "border border-blue-500/35 bg-blue-500/10 text-blue-100/80 hover:bg-blue-500/20"
              }`}
            >
              Impact
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("problem")}
              className={`rounded-full px-5 py-2 text-sm font-semibold transition ${
                activeTab === "problem"
                  ? "bg-blue-500 text-white shadow-[0_10px_24px_rgba(37,99,235,0.35)]"
                  : "border border-blue-500/35 bg-blue-500/10 text-blue-100/80 hover:bg-blue-500/20"
              }`}
            >
              Problem + Solution
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("inspiration")}
              className={`rounded-full px-5 py-2 text-sm font-semibold transition ${
                activeTab === "inspiration"
                  ? "bg-blue-500 text-white shadow-[0_10px_24px_rgba(37,99,235,0.35)]"
                  : "border border-blue-500/35 bg-blue-500/10 text-blue-100/80 hover:bg-blue-500/20"
              }`}
            >
              Inspiration
            </button>
          </div>

          {activeTab === "impact" ? (
            <div className="grid gap-5 md:grid-cols-2">
              <div className="rounded-xl border border-blue-500/25 bg-blue-600/10 p-5">
                <h3 className="text-lg font-semibold text-blue-100">Business Outcome</h3>
                <p className="mt-2 text-sm leading-relaxed text-blue-100/80">
                  FlowML helps teams reduce time spent on manual model experimentation and improves confidence in deployment decisions.
                </p>
              </div>
              <div className="rounded-xl border border-blue-500/25 bg-blue-600/10 p-5">
                <h3 className="text-lg font-semibold text-blue-100">Technical Outcome</h3>
                <p className="mt-2 text-sm leading-relaxed text-blue-100/80">
                  Stage-level visibility, reproducible outputs, and clear evaluation narratives make ML workflows easier to audit and improve.
                </p>
              </div>
            </div>
          ) : activeTab === "problem" ? (
            <div className="grid gap-5 md:grid-cols-2">
              <div className="rounded-xl border border-blue-500/25 bg-blue-600/10 p-5">
                <h3 className="text-lg font-semibold text-blue-100">The Problem</h3>
                <p className="mt-2 text-sm leading-relaxed text-blue-100/80">
                  Teams often have useful tabular data but get stuck between messy preprocessing, uncertain model choices,
                  and unclear evaluation. This slows down deployment and creates low trust in ML outcomes.
                </p>
              </div>
              <div className="rounded-xl border border-blue-500/25 bg-blue-600/10 p-5">
                <h3 className="text-lg font-semibold text-blue-100">How FlowML Solves It</h3>
                <p className="mt-2 text-sm leading-relaxed text-blue-100/80">
                  FlowML guides the full workflow from upload to delivery, with staged execution, model comparison, and
                  transparent results. It turns a fragmented ML process into one clear studio experience.
                </p>
              </div>
            </div>
          ) : (
            <div className="grid gap-5 md:grid-cols-2">
              <div className="rounded-xl border border-blue-500/25 bg-blue-600/10 p-5">
                <h3 className="text-lg font-semibold text-blue-100">Inspiration for Teams</h3>
                <p className="mt-2 text-sm leading-relaxed text-blue-100/80">
                  Use FlowML for churn prediction, demand planning, lead scoring, and risk monitoring across data-rich teams.
                </p>
              </div>
              <div className="rounded-xl border border-blue-500/25 bg-blue-600/10 p-5">
                <h3 className="text-lg font-semibold text-blue-100">Inspiration for Students</h3>
                <p className="mt-2 text-sm leading-relaxed text-blue-100/80">
                  Learn practical ML by exploring data pipelines, feature choices, and model tradeoffs through a guided interface.
                </p>
              </div>
            </div>
          )}
        </motion.section>
      </main>
    </div>
  );
};

export default Index;
