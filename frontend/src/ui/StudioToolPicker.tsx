import { useEffect, useRef, type SVGProps } from "react";
import type { StudioBffTool } from "../adk/client";

export interface StudioToolPickerProps {
  open: boolean;
  tools: StudioBffTool[];
  selectedIds: readonly string[];
  loading?: boolean;
  disabled?: boolean;
  unavailableReason?: string;
  onChange: (selectedIds: string[]) => void;
  onClose: () => void;
}

export function StudioToolsIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.75"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      <path d="M7.25 8.25a3.75 3.75 0 1 1 5.76 3.16l-5.6 5.6a1.75 1.75 0 0 1-2.48-2.48l5.6-5.6" />
      <path d="m14.8 5.2 4 4" />
      <path d="m16.2 3.8 4 4" />
    </svg>
  );
}

function CloseIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.75"
      strokeLinecap="round"
      aria-hidden="true"
      {...props}
    >
      <path d="m7 7 10 10M17 7 7 17" />
    </svg>
  );
}

export function StudioToolChips({
  tools,
  selectedIds,
  disabled,
  onChange,
}: Pick<
  StudioToolPickerProps,
  "tools" | "selectedIds" | "disabled" | "onChange"
>) {
  const selected = new Set(selectedIds);
  const items = tools.filter((tool) => selected.has(tool.id));
  if (items.length === 0) return null;

  return (
    <div className="studio-tool-chips" aria-label="已启用的本地工具">
      {items.map((tool) => (
        <span className="studio-tool-chip" key={tool.id} title={tool.description}>
          <StudioToolsIcon />
          <span>{tool.name}</span>
          <button
            type="button"
            disabled={disabled}
            aria-label={`关闭本地工具 ${tool.name}`}
            onClick={() => onChange(selectedIds.filter((id) => id !== tool.id))}
          >
            <CloseIcon />
          </button>
        </span>
      ))}
    </div>
  );
}

export function StudioToolPicker({
  open,
  tools,
  selectedIds,
  loading = false,
  disabled = false,
  unavailableReason = "",
  onChange,
  onClose,
}: StudioToolPickerProps) {
  const firstControlRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;
    firstControlRef.current?.focus();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose, open]);

  if (!open) return null;
  const selected = new Set(selectedIds);
  const controlsDisabled = disabled || loading || Boolean(unavailableReason);

  return (
    <div className="studio-tool-picker" role="dialog" aria-label="配置本地工具">
      <div className="studio-tool-picker__header">
        <div>
          <strong>本地工具</strong>
          <span>选择按会话保留，每轮独立生效</span>
        </div>
        <button
          ref={firstControlRef}
          type="button"
          className="studio-tool-picker__close"
          aria-label="关闭本地工具配置"
          onClick={onClose}
        >
          <CloseIcon />
        </button>
      </div>
      {loading ? (
        <div className="studio-tool-picker__state" role="status">
          正在读取本地工具…
        </div>
      ) : unavailableReason ? (
        <div className="studio-tool-picker__state" role="alert">
          {unavailableReason}
        </div>
      ) : tools.length === 0 ? (
        <div className="studio-tool-picker__state">当前没有可用的本地工具</div>
      ) : (
        <div className="studio-tool-picker__list">
          {tools.map((tool) => {
            const checked = selected.has(tool.id);
            return (
              <button
                key={tool.id}
                type="button"
                role="switch"
                aria-checked={checked}
                disabled={controlsDisabled}
                className="studio-tool-picker__item"
                onClick={() =>
                  onChange(
                    checked
                      ? selectedIds.filter((id) => id !== tool.id)
                      : [...selectedIds, tool.id],
                  )
                }
              >
                <span className="studio-tool-picker__copy">
                  <strong>{tool.name}</strong>
                  <span>{tool.description}</span>
                </span>
                <span className="studio-tool-switch" aria-hidden="true">
                  <span />
                </span>
              </button>
            );
          })}
        </div>
      )}
      <p className="studio-tool-picker__notice">
        工具在本机以当前登录身份执行，结果会发送给云端 Agent。
      </p>
    </div>
  );
}
