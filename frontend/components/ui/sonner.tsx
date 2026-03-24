import type { ComponentProps } from "react";
import { Toaster as Sonner } from "sonner";

type ToasterProps = ComponentProps<typeof Sonner>;

const Toaster = (props: ToasterProps) => {
  return (
    <Sonner
      position="top-right"
      richColors
      toastOptions={{
        style: {
          background: "rgba(16, 19, 31, 0.92)",
          color: "rgb(241, 245, 249)",
          border: "1px solid rgba(71, 85, 105, 0.45)",
        },
      }}
      {...props}
    />
  );
};

export { Toaster };
