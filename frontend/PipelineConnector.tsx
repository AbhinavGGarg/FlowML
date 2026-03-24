import { motion } from "framer-motion";

interface PipelineConnectorProps {
  completed: boolean;
  running: boolean;
}

const PipelineConnector = ({ completed, running }: PipelineConnectorProps) => {
  return (
    <div className="relative flex h-24 min-w-8 flex-1 items-center px-1 md:px-2">
      <div className="w-full h-[2px] bg-border/30 rounded-full" />
      <motion.div
        className="absolute left-0 top-1/2 -translate-y-1/2 h-[2px] rounded-full"
        initial={{ width: "0%" }}
        animate={{ width: completed || running ? "100%" : "0%" }}
        transition={{ duration: 0.6, ease: "easeOut" }}
        style={{
          background: completed
            ? "linear-gradient(90deg, hsl(265 80% 60%), hsl(145 70% 50%))"
            : "hsl(265 80% 60%)",
          boxShadow: completed ? "0 0 8px hsl(145 70% 50% / 0.4)" : undefined,
        }}
      />
      {completed && (
        <>
          <motion.div
            className="absolute top-1/2 -translate-y-1/2 w-1.5 h-1.5 rounded-full bg-accent"
            style={{ boxShadow: "0 0 6px hsl(145 70% 50% / 0.6)" }}
            animate={{ left: ["0%", "100%"], opacity: [0, 1, 0] }}
            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
          />
          <motion.div
            className="absolute top-1/2 -translate-y-1/2 w-1 h-1 rounded-full bg-primary"
            style={{ boxShadow: "0 0 4px hsl(265 80% 60% / 0.5)" }}
            animate={{ left: ["0%", "100%"], opacity: [0, 1, 0] }}
            transition={{ duration: 2, repeat: Infinity, ease: "linear", delay: 0.8 }}
          />
        </>
      )}
      {running && (
        <motion.div
          className="absolute top-1/2 -translate-y-1/2 w-2 h-2 rounded-full bg-primary glow-purple"
          animate={{ left: ["0%", "100%"] }}
          transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
        />
      )}
    </div>
  );
};

export default PipelineConnector;
