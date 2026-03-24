import { motion } from "framer-motion";
import { BarChart3, FileCode, FileText, Package } from "lucide-react";
import { getDownloadUrl, type MetricsResponse } from "@/lib/api";

interface ResultsPanelProps {
  isComplete: boolean;
  metrics: MetricsResponse | null;
  results: Record<string, unknown> | null;
  explanation: Record<string, unknown> | null;
}

const ResultsPanel = ({ isComplete, metrics, results, explanation }: ResultsPanelProps) => {
  if (!isComplete) return null;

  const packageReady = Boolean(results?.package_ready);
  const reportReady = Boolean(results?.report_ready);
  const hasExplanation = Boolean(explanation);

  const downloadMetrics = () => {
    if (!metrics) return;
    const blob = new Blob([JSON.stringify(metrics, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "metrics.json";
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const secondaryDownloads = [
    {
      icon: <FileCode className="h-4 w-4" />,
      label: "Download model (.pkl)",
      sublabel: results?.model_path ? "Raw model artifact" : "Unavailable",
      onClick: () => window.open(getDownloadUrl("model"), "_blank", "noopener,noreferrer"),
      disabled: !results?.model_path,
    },
    {
      icon: <FileText className="h-4 w-4" />,
      label: "Download training logs",
      sublabel: "Pipeline JSON log",
      onClick: () => window.open(getDownloadUrl("logs"), "_blank", "noopener,noreferrer"),
      disabled: false,
    },
    {
      icon: <BarChart3 className="h-4 w-4" />,
      label: "Download metrics",
      sublabel: metrics?.performance_summary || "Metrics JSON",
      onClick: downloadMetrics,
      disabled: !metrics,
    },
  ];

  return (
    <motion.div
      className="glass-card relative overflow-hidden border-primary/20 p-5"
      data-chat-context-label="Results panel"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(128,90,213,0.14),transparent_35%),radial-gradient(circle_at_bottom_left,rgba(34,197,94,0.12),transparent_30%)]" />

      <div className="relative space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.22em] text-muted-foreground">Results</p>
            <h3 className="mt-1 text-sm font-semibold text-foreground">
              <span className="text-accent">Success</span> Pipeline complete. Export results.
            </h3>
          </div>
          <span className="rounded-full border border-accent/30 bg-accent/10 px-3 py-1 text-xs text-accent">
            {hasExplanation ? "Recap ready" : "Artifacts ready"}
          </span>
        </div>

        <button
          onClick={() => window.open(getDownloadUrl("deployment-package"), "_blank", "noopener,noreferrer")}
          disabled={!packageReady}
          className="group flex w-full items-center gap-3 rounded-2xl border border-accent/30 bg-accent/10 p-4 text-left transition-all hover:border-accent/60 hover:bg-accent/20 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-accent/30 bg-accent/20 text-accent transition-colors group-hover:bg-accent/30">
            <Package className="h-5 w-5" />
          </span>
          <div className="min-w-0">
            <p className="text-sm font-semibold text-foreground">Download deployment package</p>
            <p className="mt-1 text-[11px] leading-6 text-muted-foreground">
              {packageReady
                ? "app.py · schema.json · model.pkl · Dockerfile · docker-compose.yml · README.md · report.html"
                : "Package not ready - run the full pipeline first"}
            </p>
          </div>
        </button>

        <div className="grid gap-3 md:grid-cols-2">
          <button
            onClick={() => window.open(getDownloadUrl("report"), "_blank", "noopener,noreferrer")}
            disabled={!reportReady}
            className="group flex items-center gap-3 rounded-2xl border border-border/60 bg-background/35 p-4 text-left transition-colors hover:border-accent/30 hover:bg-background/45 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <span className="flex h-9 w-9 items-center justify-center rounded-xl border border-border/60 bg-background/40 text-muted-foreground transition-colors group-hover:text-accent">
              <FileText className="h-4 w-4" />
            </span>
            <div>
              <p className="text-xs font-medium text-foreground">Open pipeline report (HTML)</p>
              <p className="mt-1 text-[10px] text-muted-foreground">Full panel-style run summary with charts</p>
            </div>
          </button>
        </div>

        <div className="grid gap-3 md:grid-cols-3">
          {secondaryDownloads.map((download) => (
            <button
              key={download.label}
              onClick={download.onClick}
              disabled={download.disabled}
              className="group flex items-center gap-3 rounded-2xl border border-border/60 bg-background/35 p-4 text-left transition-colors hover:border-accent/30 hover:bg-background/45 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <span className="flex h-9 w-9 items-center justify-center rounded-xl border border-border/60 bg-background/40 text-muted-foreground transition-colors group-hover:text-accent">
                {download.icon}
              </span>
              <div>
                <p className="text-xs font-medium text-foreground">{download.label}</p>
                <p className="mt-1 text-[10px] text-muted-foreground">{download.sublabel}</p>
              </div>
            </button>
          ))}
        </div>
      </div>
    </motion.div>
  );
};

export default ResultsPanel;
