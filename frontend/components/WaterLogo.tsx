type WaterLogoProps = {
  className?: string;
};

const WaterLogo = ({ className = "h-12 w-12" }: WaterLogoProps) => {
  return (
    <svg
      viewBox="0 0 120 120"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      role="img"
      aria-label="FlowML water logo"
    >
      <defs>
        <linearGradient id="flowmlDrop" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#60A5FA" />
          <stop offset="55%" stopColor="#2563EB" />
          <stop offset="100%" stopColor="#1D4ED8" />
        </linearGradient>
        <linearGradient id="flowmlWave" x1="0%" y1="50%" x2="100%" y2="50%">
          <stop offset="0%" stopColor="#93C5FD" />
          <stop offset="100%" stopColor="#DBEAFE" />
        </linearGradient>
      </defs>

      <path
        d="M60 9C49 27 25 48 25 71c0 20 16 36 35 36s35-16 35-36c0-23-24-44-35-62z"
        fill="url(#flowmlDrop)"
      />
      <path d="M34 68c7-6 17-8 26-6 9 2 18 0 27-6" stroke="url(#flowmlWave)" strokeWidth="5" strokeLinecap="round" fill="none" />
      <path d="M37 79c6-4 13-5 20-3 7 2 14 0 21-4" stroke="url(#flowmlWave)" strokeWidth="4" strokeLinecap="round" fill="none" opacity="0.85" />
      <circle cx="76" cy="38" r="4" fill="#E0F2FE" opacity="0.85" />
    </svg>
  );
};

export default WaterLogo;
