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
          0% { color: #a855f7; }
          33% { color: #6d28d9; }
          66% { color: #34d399; }
          100% { color: #a855f7; }
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
                  ? "'Geist Sans', sans-serif"
                  : "'Geist Mono', 'JetBrains Mono', monospace",
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
