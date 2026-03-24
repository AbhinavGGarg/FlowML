import { Link } from "react-router-dom";

const NotFound = () => {
  return (
    <div className="min-h-screen bg-background text-foreground flex items-center justify-center px-6">
      <div className="glass-card max-w-md w-full p-8 text-center space-y-4">
        <p className="text-xs font-mono text-muted-foreground">404</p>
        <h1 className="text-2xl font-semibold">Page not found</h1>
        <p className="text-sm text-muted-foreground">
          This route does not exist in the current demo app.
        </p>
        <Link
          to="/"
          className="inline-flex items-center justify-center rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition hover:opacity-90"
        >
          Back to pipeline
        </Link>
      </div>
    </div>
  );
};

export default NotFound;
