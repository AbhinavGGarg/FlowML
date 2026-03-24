import {
  forwardRef,
  type ComponentPropsWithoutRef,
  type ElementRef,
} from "react";
import * as HoverCardPrimitive from "@radix-ui/react-hover-card";

const HoverCard = HoverCardPrimitive.Root;
const HoverCardTrigger = HoverCardPrimitive.Trigger;

const HoverCardContent = forwardRef<
  ElementRef<typeof HoverCardPrimitive.Content>,
  ComponentPropsWithoutRef<typeof HoverCardPrimitive.Content>
>(({ className, align = "center", sideOffset = 8, ...props }, ref) => {
  const classes = [
    "z-50 w-64 rounded-xl border border-border bg-popover/95 p-4 text-popover-foreground shadow-xl outline-none backdrop-blur",
    className,
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <HoverCardPrimitive.Portal>
      <HoverCardPrimitive.Content
        ref={ref}
        align={align}
        sideOffset={sideOffset}
        className={classes}
        {...props}
      />
    </HoverCardPrimitive.Portal>
  );
});

HoverCardContent.displayName = HoverCardPrimitive.Content.displayName;

export { HoverCard, HoverCardContent, HoverCardTrigger };
