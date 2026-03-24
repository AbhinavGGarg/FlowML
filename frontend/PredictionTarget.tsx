import { useState } from "react";
import { motion } from "framer-motion";
import { ChevronDown, Loader2, Search } from "lucide-react";
import type { DatasetColumn } from "@/lib/api";

interface PredictionTargetProps {
  datasetLoaded: boolean;
  columns: DatasetColumn[];
  selectedColumn: string | null;
  isSaving: boolean;
  onSelect: (column: string) => Promise<void> | void;
}

const PredictionTarget = ({
  datasetLoaded,
  columns,
  selectedColumn,
  isSaving,
  onSelect,
}: PredictionTargetProps) => {
  const [searchQuery, setSearchQuery] = useState("");
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  if (!datasetLoaded) return null;

  const filtered = columns.filter((column) =>
    column.name.toLowerCase().includes(searchQuery.toLowerCase()),
  );

  const activeColumn = columns.find((column) => column.name === selectedColumn) || null;

  const handleSelect = async (column: string) => {
    await onSelect(column);
    setIsDropdownOpen(false);
    setSearchQuery("");
  };

  return (
    <motion.div
      className={`glass-card relative p-4 space-y-3 ${isDropdownOpen ? "z-30" : "z-10"}`}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2 }}
    >
      <div>
        <div>
          <h3 className="text-sm font-semibold text-foreground">What are you trying to predict?</h3>
          <p className="mt-0.5 text-[11px] text-muted-foreground">
            Pick a target column from the uploaded dataset so the agentic pipeline can analyze and train against it.
          </p>
        </div>
      </div>

      <div className="relative">
          <button
            onClick={() => setIsDropdownOpen((current) => !current)}
            disabled={columns.length === 0 || isSaving}
            className="flex w-full items-center justify-between rounded-lg bg-secondary px-3 py-2 text-sm text-foreground transition-colors hover:bg-surface-hover disabled:cursor-not-allowed disabled:opacity-50"
          >
            <span className={selectedColumn ? "text-foreground" : "text-muted-foreground"}>
              {selectedColumn || "Select column..."}
            </span>
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          </button>

          {isDropdownOpen && (
            <motion.div
              className="absolute top-full z-40 mt-1 w-full overflow-hidden rounded-lg border border-border/50 glass-card"
              initial={{ opacity: 0, y: -5 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <div className="border-b border-border/30 p-2">
                <div className="flex items-center gap-2 rounded-md bg-secondary px-2 py-1.5">
                  <Search className="h-3.5 w-3.5 text-muted-foreground" />
                  <input
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search columns..."
                    className="flex-1 bg-transparent text-sm text-foreground placeholder:text-muted-foreground focus:outline-none"
                    autoFocus
                  />
                </div>
              </div>
              <div className="scrollbar-thin max-h-48 overflow-y-auto p-1">
                {filtered.map((column) => (
                  <button
                    key={column.name}
                    onClick={() => void handleSelect(column.name)}
                    className="w-full rounded-md px-3 py-2 text-left transition-colors hover:bg-secondary"
                  >
                    <div className="flex items-center justify-between gap-3">
                      <span className="text-sm font-mono text-foreground">{column.name}</span>
                      <span className="text-[10px] text-muted-foreground">
                        {column.dtype}
                        {column.missing_pct > 0 ? ` • ${(column.missing_pct * 100).toFixed(1)}% missing` : ""}
                      </span>
                    </div>
                  </button>
                ))}
                {filtered.length === 0 && (
                  <p className="px-3 py-2 text-xs text-muted-foreground">No columns found</p>
                )}
              </div>
            </motion.div>
          )}
      </div>

      <div className="flex flex-wrap items-center gap-2 text-[11px] font-mono">
        {selectedColumn ? (
          <>
            <span className="text-muted-foreground">Target:</span>
            <span className="rounded-md bg-accent/10 px-2 py-0.5 font-semibold text-accent">{selectedColumn}</span>
            {activeColumn && (
              <span className="rounded-md bg-secondary px-2 py-0.5 text-muted-foreground">
                {activeColumn.is_numeric ? "numeric" : "categorical"} • {activeColumn.dtype}
              </span>
            )}
            {isSaving && (
              <span className="flex items-center gap-1 text-primary">
                <Loader2 className="h-3 w-3 animate-spin" />
                saving target
              </span>
            )}
          </>
        ) : (
          <span className="text-muted-foreground">{columns.length} dataset columns available for selection</span>
        )}
      </div>
    </motion.div>
  );
};

export default PredictionTarget;
