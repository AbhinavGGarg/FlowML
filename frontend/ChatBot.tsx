import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Bot, MessageCircle, Quote, Send, Sparkles, User, X } from "lucide-react";
import {
  queryChat,
  type ChatSelectionContext,
  type MetricsResponse,
  type TaskType,
} from "@/lib/api";

interface Message {
  role: "user" | "assistant";
  content: string;
  source?: "llm" | "unavailable" | "structured";
}

interface ChatBotProps {
  datasetName: string | null;
  targetColumn: string | null;
  taskType: TaskType;
  activeStageId: string | null;
  stageLogs: Record<string, string[]>;
  metrics: MetricsResponse | null;
  onRevisionRerunStart?: (rerunFromStage: string) => void;
  onRevisionRerunComplete?: () => void | Promise<void>;
  onPipelineRefresh?: () => void | Promise<void>;
}

interface FloatingSelection extends ChatSelectionContext {
  x: number;
  y: number;
}

const buildChatHistory = (messages: Message[]) => {
  const sanitized: Array<{ role: "user" | "assistant"; content: string }> = [];

  for (const message of messages) {
    if (message.role === "assistant" && message.source === "unavailable") {
      if (sanitized.length > 0 && sanitized[sanitized.length - 1].role === "user") {
        sanitized.pop();
      }
      continue;
    }

    sanitized.push({
      role: message.role,
      content: message.content,
    });
  }

  return sanitized.slice(-8);
};

const ChatBot = ({
  datasetName,
  targetColumn,
  taskType,
  activeStageId,
  stageLogs,
  metrics,
  onRevisionRerunStart,
  onRevisionRerunComplete,
  onPipelineRefresh,
}: ChatBotProps) => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content: "I can explain your dataset, each pipeline stage, the selected model, and the final results.",
    },
  ]);
  const [input, setInput] = useState("");
  const [isThinking, setIsThinking] = useState(false);
  const [pinnedContext, setPinnedContext] = useState<ChatSelectionContext | null>(null);
  const [floatingSelection, setFloatingSelection] = useState<FloatingSelection | null>(null);
  const [pendingRevisionPlan, setPendingRevisionPlan] = useState<Record<string, unknown> | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const contextSummary = useMemo(() => {
    const parts: string[] = [];
    if (datasetName) parts.push(`Dataset: ${datasetName}`);
    if (targetColumn) parts.push(`Target: ${targetColumn}`);
    parts.push(`Task: ${taskType}`);
    if (activeStageId) {
      parts.push(`Active stage: ${activeStageId}`);
    } else if (metrics?.model_name) {
      parts.push(`Model: ${metrics.model_name}`);
    }
    if (taskType === "regression" && metrics?.r2 != null && metrics?.rmse != null) {
      parts.push(`R2 = ${metrics.r2.toFixed(3)}`);
      parts.push(`RMSE = ${metrics.rmse.toFixed(3)}`);
    } else if (taskType === "classification" && metrics?.accuracy != null && metrics?.f1 != null) {
      parts.push(`Accuracy = ${(metrics.accuracy * 100).toFixed(1)}%`);
      parts.push(`F1 = ${(metrics.f1 * 100).toFixed(1)}%`);
    }
    if (metrics?.deployment_decision) parts.push(`Decision: ${metrics.deployment_decision}`);
    return parts.join(" | ");
  }, [activeStageId, datasetName, metrics, targetColumn, taskType]);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages, isThinking]);

  useEffect(() => {
    setMessages((current) => {
      const next = [...current];
      next[0] = {
        role: "assistant",
        content:
          contextSummary ||
          "I can explain your dataset, what each stage did, why the model was chosen, and how to read the final metrics.",
      };
      return next;
    });
  }, [contextSummary]);

  useEffect(() => {
    const updateSelection = () => {
      window.requestAnimationFrame(() => {
        const next = readHighlightedContext(panelRef.current);
        setFloatingSelection(next);
      });
    };

    const clearFloatingSelection = () => {
      const selection = window.getSelection();
      if (!selection || selection.isCollapsed || !selection.toString().trim()) {
        setFloatingSelection(null);
      }
    };

    document.addEventListener("mouseup", updateSelection);
    document.addEventListener("keyup", updateSelection);
    document.addEventListener("selectionchange", clearFloatingSelection);
    window.addEventListener("scroll", clearFloatingSelection, true);

    return () => {
      document.removeEventListener("mouseup", updateSelection);
      document.removeEventListener("keyup", updateSelection);
      document.removeEventListener("selectionchange", clearFloatingSelection);
      window.removeEventListener("scroll", clearFloatingSelection, true);
    };
  }, []);

  const handleUseSelection = () => {
    if (!floatingSelection) return;

    const nextContext = {
      text: floatingSelection.text,
      source_label: floatingSelection.source_label,
      surrounding_text: floatingSelection.surrounding_text,
    };
    setPinnedContext(nextContext);
    setFloatingSelection(null);
    setIsOpen(true);
    window.getSelection()?.removeAllRanges();
    window.setTimeout(() => inputRef.current?.focus(), 80);
  };

  const isApplyConfirmation = (value: string) => {
    const normalized = value.trim().toLowerCase();
    return [
      "apply",
      "apply it",
      "proceed",
      "continue",
      "go ahead",
      "do it",
      "do that",
      "yes",
      "yes please",
      "sure",
      "yes, apply it",
    ].includes(normalized);
  };

  const isDirectRevisionCommand = (value: string) => {
    const normalized = value.trim().toLowerCase();
    if (!normalized) return false;
    if (isApplyConfirmation(normalized)) return true;

    const explanatoryPrefixes = [
      "why ",
      "how ",
      "what ",
      "which ",
      "can you explain",
      "could you explain",
      "explain why",
      "tell me why",
    ];
    if (explanatoryPrefixes.some((prefix) => normalized.startsWith(prefix))) {
      return false;
    }

    const directChangeMarkers = [
      "run without",
      "train without",
      "retrain without",
      "remove ",
      "drop ",
      "exclude ",
      "add ",
      "include ",
      "switch to",
      "use ",
      "undo ",
      "revert ",
      "go back",
      "make the model",
      "change the",
    ];
    const preferenceMarkers = ["i want to", "i would like to", "please", "let's", "lets"];

    if (directChangeMarkers.some((marker) => normalized.includes(marker))) {
      return true;
    }

    return (
      preferenceMarkers.some((marker) => normalized.includes(marker)) &&
      ["without ", "remove ", "drop ", "exclude ", "add ", "include ", "switch "].some((marker) =>
        normalized.includes(marker),
      )
    );
  };

  const runChatRequest = async ({
    prompt,
    mode,
    userMessage = prompt,
  }: {
    prompt: string;
    mode: "suggest" | "apply";
    userMessage?: string;
  }) => {
    const trimmed = prompt.trim();
    if (!trimmed) return;

    const userMsg: Message = { role: "user", content: userMessage.trim() };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsThinking(true);

    const rerunFromStage =
      mode === "apply" && pendingRevisionPlan
        ? String(pendingRevisionPlan.rerun_from_stage ?? "")
        : "";
    const shouldTrackRerun = mode === "apply" && Boolean(rerunFromStage);
    let refreshInterval: number | null = null;
    let didApplyRevision = false;

    if (shouldTrackRerun) {
      onRevisionRerunStart?.(rerunFromStage);
      if (onPipelineRefresh) {
        refreshInterval = window.setInterval(() => {
          void onPipelineRefresh();
        }, 250);
      }
    }

    try {
      const history = buildChatHistory(messages);
      const response = await queryChat(trimmed, history, pinnedContext, mode);
      const revision = response.revision as Record<string, unknown> | null | undefined;
      didApplyRevision = revision?.applied === true;

      if (revision?.mode === "suggest" && revision?.applied === false) {
        const plan = revision.plan as Record<string, unknown> | undefined;
        setPendingRevisionPlan(plan ?? null);
      } else if (revision?.applied === true) {
        setPendingRevisionPlan(null);
      }

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: response.answer,
          source: response.response_mode ?? (response.llm_used ? "llm" : "unavailable"),
        },
      ]);
    } catch (error) {
      const message =
        error instanceof Error
          ? buildChatErrorMessage(error.message)
          : buildEmergencyFallback({
              datasetName,
              targetColumn,
              taskType,
              activeStageId,
              stageLogs,
              metrics,
              pinnedContext,
            });
      setMessages((prev) => [...prev, { role: "assistant", content: message, source: "unavailable" }]);
    } finally {
      if (refreshInterval !== null) {
        window.clearInterval(refreshInterval);
      }
      if (shouldTrackRerun || didApplyRevision) {
        await onPipelineRefresh?.();
        await onRevisionRerunComplete?.();
      }
      setIsThinking(false);
    }
  };

  const handleSend = async () => {
    if (!input.trim()) return;
    const nextMode =
      (pendingRevisionPlan && isApplyConfirmation(input)) || isDirectRevisionCommand(input)
        ? "apply"
        : "suggest";
    await runChatRequest({ prompt: input, mode: nextMode });
  };

  const handleApplyPendingRevision = async () => {
    if (!pendingRevisionPlan || isThinking) return;
    await runChatRequest({
      prompt: "apply it",
      mode: "apply",
      userMessage: "Apply the suggested pipeline revision.",
    });
  };

  return (
    <>
      <AnimatePresence>
        {floatingSelection && (
          <motion.button
            className="fixed z-50 flex items-center gap-2 rounded-full border border-primary/40 bg-background/95 px-3 py-2 text-xs font-medium text-foreground shadow-[0_12px_30px_rgba(0,0,0,0.35)] backdrop-blur"
            style={{
              left: floatingSelection.x,
              top: floatingSelection.y,
              transform: "translate(-50%, -100%)",
            }}
            initial={{ opacity: 0, y: 8, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 8, scale: 0.96 }}
            onClick={handleUseSelection}
          >
            <Sparkles className="h-3.5 w-3.5 text-primary" />
            Ask FlowML about this
          </motion.button>
        )}
      </AnimatePresence>

      <motion.button
        className="glow-purple fixed bottom-6 left-6 z-50 flex h-14 w-14 items-center justify-center rounded-full bg-primary transition-transform hover:scale-110"
        onClick={() => setIsOpen((current) => !current)}
        whileTap={{ scale: 0.9 }}
      >
        {isOpen ? <X className="h-6 w-6 text-primary-foreground" /> : <MessageCircle className="h-6 w-6 text-primary-foreground" />}
      </motion.button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            ref={panelRef}
            className="glass-card fixed bottom-24 left-6 z-50 flex h-[32rem] w-80 flex-col overflow-hidden rounded-xl border border-border/50 sm:w-[26rem]"
            initial={{ opacity: 0, y: 20, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.9 }}
            transition={{ type: "spring", damping: 25 }}
            data-chat-context-label="Chat assistant"
          >
            <div className="flex items-center gap-2 border-b border-border/30 px-4 py-3">
              <Bot className="h-4 w-4 text-accent" />
              <span className="text-sm font-semibold text-foreground">FlowML Assistant</span>
              <span className="ml-auto text-[10px] font-mono text-accent">online</span>
            </div>

            {pinnedContext && (
              <div className="border-b border-border/30 px-3 py-3">
                <div className="rounded-xl border border-primary/20 bg-primary/8 p-3">
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-center gap-2 text-[11px] font-medium uppercase tracking-[0.18em] text-primary">
                      <Quote className="h-3.5 w-3.5" />
                      Highlighted Context
                    </div>
                    <button
                      onClick={() => setPinnedContext(null)}
                      className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>
                  <p className="mt-2 text-xs leading-6 text-foreground">{pinnedContext.text}</p>
                  {pinnedContext.source_label && (
                    <p className="mt-2 text-[11px] text-muted-foreground">Source: {pinnedContext.source_label}</p>
                  )}
                </div>
              </div>
            )}

            <div ref={scrollRef} className="scrollbar-thin flex-1 space-y-3 overflow-y-auto p-3">
              {messages.map((message, index) => (
                <motion.div
                  key={`${message.role}-${index}`}
                  initial={{ opacity: 0, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`flex gap-2 ${message.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  {message.role === "assistant" && (
                    <div className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-accent/20">
                      <Bot className="h-3 w-3 text-accent" />
                    </div>
                  )}
                  <div
                    className={`max-w-[82%] rounded-xl px-3 py-2 text-[12px] leading-relaxed ${
                      message.role === "user" ? "bg-primary/20 text-foreground" : "bg-secondary text-secondary-foreground"
                    }`}
                  >
                    {renderChatContent(message.content)}
                    {message.role === "assistant" && message.source && (
                      <div className="mt-2 text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
                        {message.source === "llm"
                          ? "LLM response"
                          : message.source === "unavailable"
                            ? "LLM unavailable"
                            : "Structured response"}
                      </div>
                    )}
                  </div>
                  {message.role === "user" && (
                    <div className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary/20">
                      <User className="h-3 w-3 text-primary" />
                    </div>
                  )}
                </motion.div>
              ))}
              {isThinking && (
                <motion.div
                  initial={{ opacity: 0, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex justify-start gap-2"
                >
                  <div className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-accent/20">
                    <Bot className="h-3 w-3 text-accent" />
                  </div>
                  <div className="rounded-xl bg-secondary px-3 py-2 text-[12px] text-secondary-foreground">
                    Thinking through your run...
                  </div>
                </motion.div>
              )}
            </div>

            <div className="border-t border-border/30 p-3">
              {pendingRevisionPlan && (
                <div className="mb-3 rounded-xl border border-primary/20 bg-primary/8 p-3">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="text-[11px] font-medium uppercase tracking-[0.18em] text-primary">
                        Suggested Revision
                      </div>
                      <p className="mt-1 text-xs leading-5 text-foreground">
                        {String(pendingRevisionPlan.reason ?? "A pipeline revision is ready to apply.")}
                      </p>
                      {pendingRevisionPlan.rerun_from_stage && (
                        <p className="mt-1 text-[11px] text-muted-foreground">
                          Reruns from: {String(pendingRevisionPlan.rerun_from_stage)}
                        </p>
                      )}
                    </div>
                    <button
                      onClick={() => setPendingRevisionPlan(null)}
                      className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>
                  <button
                    onClick={() => void handleApplyPendingRevision()}
                    disabled={isThinking}
                    className="mt-3 rounded-lg bg-primary px-3 py-2 text-xs font-medium text-primary-foreground transition-colors hover:bg-primary/85 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    Apply revision
                  </button>
                </div>
              )}
              <div className="mb-2 text-[11px] text-muted-foreground">
                {pinnedContext
                  ? "Your next question will include the highlighted snippet as extra context."
                  : "Highlight text anywhere in the app to bring it into the chat."}
              </div>
              <div className="flex gap-2">
                <input
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && void handleSend()}
                  placeholder={pinnedContext ? "Ask about the highlighted text..." : "Ask about this dataset or stage..."}
                  className="flex-1 rounded-lg bg-secondary px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary/50"
                />
                <button
                  onClick={() => void handleSend()}
                  className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary transition-colors hover:bg-primary/80"
                >
                  <Send className="h-4 w-4 text-primary-foreground" />
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};

const readHighlightedContext = (panelElement: HTMLDivElement | null): FloatingSelection | null => {
  const selection = window.getSelection();
  if (!selection || selection.rangeCount === 0 || selection.isCollapsed) return null;

  const text = selection.toString().replace(/\s+/g, " ").trim();
  if (text.length < 3) return null;

  const range = selection.getRangeAt(0);
  const rect = range.getBoundingClientRect();
  const commonNode = range.commonAncestorContainer;
  const commonElement =
    commonNode.nodeType === Node.ELEMENT_NODE ? (commonNode as Element) : commonNode.parentElement;

  if (!commonElement) return null;
  if (panelElement?.contains(commonElement)) return null;
  if (commonElement.closest("input, textarea, button")) return null;

  const sourceLabel = resolveSourceLabel(commonElement);
  const surroundingText = extractSurroundingText(commonElement, text);
  const cappedText = text.length > 280 ? `${text.slice(0, 277)}...` : text;

  return {
    text: cappedText,
    source_label: sourceLabel,
    surrounding_text: surroundingText,
    x: clamp(rect.left + rect.width / 2, 120, window.innerWidth - 120),
    y: clamp(rect.top - 10, 70, window.innerHeight - 140),
  };
};

const resolveSourceLabel = (element: Element) => {
  const labeledAncestor = element.closest<HTMLElement>("[data-chat-context-label]");
  if (labeledAncestor?.dataset.chatContextLabel) {
    return labeledAncestor.dataset.chatContextLabel;
  }

  const section = element.closest("section, article, div");
  const heading = section?.querySelector("h1, h2, h3, h4");
  return heading?.textContent?.trim() || "current view";
};

const extractSurroundingText = (element: Element, selectedText: string) => {
  const container =
    element.closest<HTMLElement>("[data-chat-context-label]") ||
    element.closest<HTMLElement>("section, article, div, p, li, td, th");
  const content = container?.textContent?.replace(/\s+/g, " ").trim();
  if (!content) return undefined;

  if (content.length <= 320) return content;

  const needle = selectedText.slice(0, 40);
  const index = content.indexOf(needle);
  if (index === -1) return `${content.slice(0, 317)}...`;

  const start = Math.max(0, index - 120);
  const end = Math.min(content.length, index + selectedText.length + 120);
  const prefix = start > 0 ? "..." : "";
  const suffix = end < content.length ? "..." : "";
  return `${prefix}${content.slice(start, end)}${suffix}`;
};

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

const renderChatContent = (content: string) => {
  const blocks = content.split(/\n\s*\n/).map((block) => block.trim()).filter(Boolean);

  return (
    <div className="space-y-3">
      {blocks.map((block, blockIndex) => {
        const lines = block.split("\n").map((line) => line.trim()).filter(Boolean);
        const isBulletList = lines.length > 0 && lines.every((line) => line.startsWith("- "));

        if (isBulletList) {
          return (
            <ul key={`block-${blockIndex}`} className="space-y-1.5 pl-4">
              {lines.map((line, lineIndex) => (
                <li key={`bullet-${blockIndex}-${lineIndex}`} className="list-disc marker:text-primary">
                  {renderInlineFormatting(line.slice(2))}
                </li>
              ))}
            </ul>
          );
        }

        return (
          <div key={`block-${blockIndex}`} className="space-y-1.5">
            {lines.map((line, lineIndex) => {
              const isSectionLabel =
                line.endsWith(":") &&
                !line.startsWith("- ") &&
                line.length <= 48;

              if (isSectionLabel) {
                return (
                  <p
                    key={`line-${blockIndex}-${lineIndex}`}
                    className="pt-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-primary"
                  >
                    {renderInlineFormatting(line.slice(0, -1))}
                  </p>
                );
              }

              return (
                <p key={`line-${blockIndex}-${lineIndex}`} className="leading-6">
                  {renderInlineFormatting(line)}
                </p>
              );
            })}
          </div>
        );
      })}
    </div>
  );
};

const renderInlineFormatting = (text: string) => {
  const segments = text.split(/(\*\*.*?\*\*|`.*?`)/g).filter(Boolean);

  return segments.map((segment, index) => {
    if (segment.startsWith("**") && segment.endsWith("**")) {
      return (
        <strong key={`${segment}-${index}`} className="font-semibold text-foreground">
          {segment.slice(2, -2)}
        </strong>
      );
    }

    if (segment.startsWith("`") && segment.endsWith("`")) {
      return (
        <code
          key={`${segment}-${index}`}
          className="rounded bg-background/60 px-1.5 py-0.5 font-mono text-[11px] text-accent"
        >
          {segment.slice(1, -1)}
        </code>
      );
    }

    return <Fragment key={`${segment}-${index}`}>{segment}</Fragment>;
  });
};

const buildEmergencyFallback = (context: {
  datasetName: string | null;
  targetColumn: string | null;
  taskType: TaskType;
  activeStageId: string | null;
  stageLogs: Record<string, string[]>;
  metrics: MetricsResponse | null;
  pinnedContext: ChatSelectionContext | null;
}) => {
  const lines = [
    context.datasetName ? `Dataset: ${context.datasetName}` : "No dataset loaded yet.",
    context.targetColumn ? `Target: ${context.targetColumn}` : "No target selected yet.",
    `Task: ${context.taskType}`,
  ];
  if (context.activeStageId) {
    lines.push(`Active stage: ${context.activeStageId}`);
  }
  if (context.pinnedContext?.text) {
    lines.push(`Highlighted text: "${context.pinnedContext.text}"`);
    if (context.pinnedContext.source_label) {
      lines.push(`Source area: ${context.pinnedContext.source_label}`);
    }
  }
  if (context.taskType === "regression" && context.metrics?.r2 != null && context.metrics?.rmse != null) {
    lines.push(
      `Current regression metrics: R2 = ${context.metrics.r2.toFixed(3)}, RMSE = ${context.metrics.rmse.toFixed(3)}`,
    );
  }
  if (context.taskType === "classification" && context.metrics?.accuracy != null && context.metrics?.f1 != null) {
    lines.push(
      `Current classification metrics: Accuracy = ${(context.metrics.accuracy * 100).toFixed(1)}%, F1 = ${(context.metrics.f1 * 100).toFixed(1)}%`,
    );
  }
  return `${lines.join("\n")}\nThe full chat assistant is temporarily unavailable, but the pipeline context is still loaded.`;
};

const buildChatErrorMessage = (detail: string) => {
  const normalized = detail.trim();

  if (!normalized) {
    return "The chat request failed before the assistant could answer. Please try again.";
  }

  if (normalized.toLowerCase().includes("timed out")) {
    return "That took too long, so I stopped waiting for the chat model. Please try again.";
  }

  return normalized;
};

export default ChatBot;
