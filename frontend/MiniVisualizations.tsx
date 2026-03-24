import type { ReactNode } from "react";

const MiniHeatmap = () => (
  <div className="flex gap-0.5">
    {[0.8, 0.4, -0.2, 0.6, 1.0].map((v, i) => (
      <div
        key={i}
        className="w-4 h-4 rounded-sm"
        style={{ backgroundColor: `hsl(${v > 0 ? 265 : 145} 80% 60% / ${Math.abs(v) * 0.5})` }}
      />
    ))}
  </div>
);

const MiniBar = () => (
  <div className="space-y-1">
    {[0.9, 0.7, 0.5].map((v, i) => (
      <div key={i} className="h-2 bg-secondary rounded-sm overflow-hidden">
        <div className="h-full rounded-sm" style={{ width: `${v * 100}%`, background: "linear-gradient(90deg, hsl(265 80% 60%), hsl(145 70% 50%))" }} />
      </div>
    ))}
  </div>
);

const MiniCurve = () => (
  <svg viewBox="0 0 60 30" className="w-full h-6">
    <polyline points="0,28 10,22 20,16 30,10 40,6 50,4 60,3" fill="none" stroke="hsl(265 80% 60%)" strokeWidth="1.5" />
    <polyline points="0,29 10,24 20,18 30,14 40,12 50,11 60,12" fill="none" stroke="hsl(145 70% 50%)" strokeWidth="1.5" strokeDasharray="2" />
  </svg>
);

const MiniMetrics = () => (
  <div className="flex gap-2 text-[9px] font-mono">
    <span className="text-accent">93.4%</span>
    <span className="text-primary">F1: 93.3%</span>
  </div>
);

export const MiniVisualization = ({ vizType }: { vizType: string }) => {
  const map: Record<string, ReactNode> = {
    heatmap: <MiniHeatmap />,
    barChart: <MiniBar />,
    lossCurve: <MiniCurve />,
    table: <MiniBar />,
    confusionMatrix: <MiniMetrics />,
    metrics: <MiniMetrics />,
  };
  return <>{map[vizType] || null}</>;
};
