import { motion } from "framer-motion";
import { AlertTriangle, Columns3, Database, Hash, Rows3, Type } from "lucide-react";
import type { DatasetColumn, DatasetSummary as DatasetSummaryData } from "@/lib/api";

interface DatasetSummaryProps {
  summary: DatasetSummaryData | null;
  columns: DatasetColumn[];
  targetColumn: string | null;
}

const DatasetSummary = ({ summary, columns, targetColumn }: DatasetSummaryProps) => {
  if (!summary) return null;

  const numericCount = columns.filter((column) => column.is_numeric).length;
  const categoricalCount = columns.length - numericCount;
  const averageMissing =
    columns.length > 0
      ? columns.reduce((total, column) => total + column.missing_pct, 0) / columns.length
      : 0;

  const stats = [
    { icon: <Rows3 className="h-4 w-4" />, label: "Rows", value: summary.rows.toLocaleString() },
    { icon: <Columns3 className="h-4 w-4" />, label: "Columns", value: summary.columns.toString() },
    { icon: <AlertTriangle className="h-4 w-4" />, label: "Avg Missing", value: `${(averageMissing * 100).toFixed(1)}%` },
    { icon: <Hash className="h-4 w-4" />, label: "Numeric", value: numericCount.toString() },
    { icon: <Type className="h-4 w-4" />, label: "Categorical", value: categoricalCount.toString() },
  ];

  return (
    <motion.div
      className="glass-card relative z-0 overflow-hidden border-primary/20 p-5"
      data-chat-context-label="Dataset summary"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 }}
    >
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(128,90,213,0.14),transparent_35%),radial-gradient(circle_at_bottom_left,rgba(34,197,94,0.12),transparent_30%)]" />

      <div className="relative space-y-5">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.22em] text-muted-foreground">Dataset Overview</p>
            <p className="mt-1 flex items-center gap-2 text-sm text-secondary-foreground">
              <Database className="h-4 w-4 text-foreground/80" />
              {summary.filename}
            </p>
          </div>
          {targetColumn && (
            <span className="self-start rounded-full border border-accent/30 bg-accent/10 px-3 py-1 text-xs text-accent">
              Target column: <span className="font-mono">{targetColumn}</span>
            </span>
          )}
        </div>

        <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
          {stats.map((stat) => (
            <div key={stat.label} className="rounded-2xl border border-border/60 bg-background/35 p-4">
              <span className="text-foreground/80">{stat.icon}</span>
              <p className="mt-3 font-mono text-2xl font-semibold text-foreground">{stat.value}</p>
              <p className="mt-1 text-[11px] uppercase tracking-[0.2em] text-muted-foreground">{stat.label}</p>
            </div>
          ))}
        </div>

        <div className="space-y-2">
          <p className="text-xs font-semibold uppercase tracking-[0.22em] text-muted-foreground">Columns in Play</p>
          <div className="flex flex-wrap gap-2">
            {columns.slice(0, 12).map((column) => (
              <span
                key={column.name}
                className={`rounded-full border px-3 py-2 text-sm ${
                  column.name === targetColumn
                    ? "border-accent/30 bg-accent/10 text-accent"
                    : "border-border/60 bg-background/35 text-foreground"
                }`}
              >
                {column.name}
              </span>
            ))}
            {columns.length > 12 && (
              <span className="rounded-full border border-border/60 bg-background/35 px-3 py-2 text-sm text-muted-foreground">
                +{columns.length - 12} more
              </span>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default DatasetSummary;
