import { motion } from "framer-motion";

const TitleComponent = () => {
  const brand = "FlowML";
  const extension = "";
  const allChars = (brand + extension).split("");

  const container = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.04, delayChildren: 0.1 },
    },
  };

  const child = {
    visible: {
      opacity: 1,
      y: 0,
      transition: { type: "spring" as const, damping: 12, stiffness: 200 },
    },
    hidden: { opacity: 0, y: 4 },
  };

  return (
    <>
      <style>{`
        @keyframes color-flow {
          0% { color: #60a5fa; }
          33% { color: #3b82f6; }
          66% { color: #38bdf8; }
          100% { color: #60a5fa; }
        }
      `}</style>
      <motion.div
        className="flex items-center select-none cursor-default"
        variants={container}
        initial="hidden"
        animate="visible"
      >
        {allChars.map((char, index) => (
          <motion.span
            key={index}
            variants={child}
            style={{
              fontFamily:
                index < brand.length
                  ? "'Sora', 'Geist Sans', sans-serif"
                  : "'IBM Plex Mono', 'Geist Mono', monospace",
              letterSpacing: index < brand.length ? "-0.03em" : "0",
              fontWeight: index < brand.length ? 600 : 400,
              fontSize: "clamp(4.25rem, 9vw, 6.25rem)",
              lineHeight: 1,
              animation: "color-flow 3s ease-in-out infinite",
              animationDelay: `${index * 0.3}s`,
            }}
          >
            {char}
          </motion.span>
        ))}
      </motion.div>
    </>
  );
};

export default TitleComponent;
