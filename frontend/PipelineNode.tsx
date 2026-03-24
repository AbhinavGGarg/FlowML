import { motion } from "framer-motion";
import {
  AlertCircle,
  Bot,
  Check,
  Compass,
  Droplets,
  FlaskConical,
  Layers,
  Loader2,
  PackageOpen,
  SearchCheck,
} from "lucide-react";
import { StageStatus, PipelineStage } from "@/data/pipelineStages";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import { MiniVisualization } from "./MiniVisualizations";

interface PipelineNodeProps {
  stage: PipelineStage;
  status: StageStatus;
  index: number;
  onClick: () => void;
}

const PipelineNode = ({ stage, status, index, onClick }: PipelineNodeProps) => {
  const isClickable = status === "completed" || status === "failed";

  const stageIconMap: Record<string, React.ComponentType<{ className?: string; strokeWidth?: number }>> = {
    analysis: SearchCheck,
    preprocessing: Droplets,
    features: Layers,
    model_selection: Compass,
    training: Bot,
    evaluation: FlaskConical,
    results: PackageOpen,
    loss: Layers,
  };

  const renderStageIcon = (className = "h-8 w-8") => {
    const Icon = stageIconMap[stage.id] || Droplets;
    return <Icon className={`${className} text-blue-200`} strokeWidth={2.1} />;
  };

  const nodeContent = (
    <motion.button
      onClick={isClickable ? onClick : undefined}
      className={`relative flex flex-col items-center gap-2 group ${isClickable ? "cursor-pointer" : "cursor-default"}`}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1, duration: 0.4 }}
    >
      <motion.div
        className={`relative h-24 w-24 rounded-[1.75rem] flex items-center justify-center text-2xl transition-all duration-500 border
          ${status === "waiting" ? "bg-secondary/85 border-border/50" : ""}
          ${status === "running" ? "bg-secondary border-primary/60 glow-purple" : ""}
          ${status === "completed" ? "bg-blue-500/15 border-blue-300/70 glow-green-sm group-hover:glow-green" : ""}
          ${status === "failed" ? "bg-destructive/10 border-destructive/70" : ""}
        `}
        whileHover={isClickable ? { scale: 1.1, y: -4 } : {}}
        whileTap={isClickable ? { scale: 0.95 } : {}}
      >
        {status === "running" && (
          <motion.div
            className="absolute inset-0 rounded-2xl border-2 border-primary/40"
            animate={{ scale: [1, 1.2, 1], opacity: [0.5, 0, 0.5] }}
            transition={{ duration: 2, repeat: Infinity }}
          />
        )}

        {status === "running" ? (
          <Loader2 className="w-6 h-6 text-primary animate-spin" />
        ) : status === "failed" ? (
          <AlertCircle className="w-6 h-6 text-destructive" />
        ) : status === "completed" ? (
          <div className="relative">
            <div>{renderStageIcon()}</div>
            <motion.div
              className="absolute -top-1 -right-2 w-5 h-5 rounded-full bg-blue-400 flex items-center justify-center"
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ type: "spring", delay: 0.2 }}
            >
              <Check className="w-3 h-3 text-slate-900" strokeWidth={3} />
            </motion.div>
          </div>
        ) : (
          renderStageIcon()
        )}
      </motion.div>

      <span
        className={`text-sm font-medium transition-colors duration-500 max-w-[110px] text-center leading-tight
          ${status === "waiting" ? "text-muted-foreground/40" : ""}
          ${status === "running" ? "text-primary" : ""}
          ${status === "completed" ? "text-foreground group-hover:text-blue-300" : ""}
          ${status === "failed" ? "text-destructive" : ""}
        `}
      >
        {stage.label}
      </span>
    </motion.button>
  );

  if (status === "waiting") return nodeContent;

  return (
    <HoverCard openDelay={200} closeDelay={100}>
      <HoverCardTrigger asChild>{nodeContent}</HoverCardTrigger>
      <HoverCardContent className="w-72 glass-card border-border/50 p-0 overflow-hidden" side="bottom" sideOffset={12}>
        <div className="p-3 border-b border-border/30">
          <div className="flex items-center gap-2 mb-1">
            <div className="flex h-8 w-8 items-center justify-center">{renderStageIcon("h-6 w-6")}</div>
            <span className="text-sm font-semibold text-foreground">{stage.label}</span>
            {status === "completed" && (
              <span className="ml-auto text-[10px] font-mono text-accent">✓ done</span>
            )}
            {status === "running" && (
              <span className="ml-auto text-[10px] font-mono text-primary animate-pulse">running...</span>
            )}
            {status === "failed" && (
              <span className="ml-auto text-[10px] font-mono text-destructive">failed</span>
            )}
          </div>
          <p className="text-[11px] text-muted-foreground">{stage.description}</p>
        </div>
        <div className="px-3 py-2">
          <ul className="space-y-1">
            {stage.tooltipPoints.map((point, i) => (
              <li key={i} className="text-[11px] text-secondary-foreground flex items-center gap-1.5">
                <span className="w-1 h-1 rounded-full bg-accent shrink-0" />
                {point}
              </li>
            ))}
          </ul>
        </div>
        {status === "completed" && (
          <div className="px-3 pb-3">
            <div className="rounded-lg bg-secondary/50 p-2 scale-[0.85] origin-top-left">
              <MiniVisualization vizType={stage.vizType} />
            </div>
          </div>
        )}
      </HoverCardContent>
    </HoverCard>
  );
};

export default PipelineNode;
