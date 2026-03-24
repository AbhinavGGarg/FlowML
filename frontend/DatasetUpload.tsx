import { useState } from "react";
import { motion } from "framer-motion";
import { FileSpreadsheet, Loader2, Upload } from "lucide-react";

interface DatasetUploadProps {
  fileName?: string | null;
  isUploading: boolean;
  error?: string | null;
  onUpload: (file: File) => Promise<void> | void;
}

const DatasetUpload = ({ fileName, isUploading, error, onUpload }: DatasetUploadProps) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleFile = async (file: File) => {
    await onUpload(file);
  };

  return (
    <motion.div
      className={`glass-card p-4 border-dashed border-2 transition-colors cursor-pointer ${
        isDragging ? "border-accent bg-accent/5" : "border-border/50 hover:border-primary/50"
      }`}
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files[0];
        if (file) {
          void handleFile(file);
        }
      }}
      onClick={() => {
        if (isUploading) return;
        const input = document.createElement("input");
        input.type = "file";
        input.accept = ".csv,.json,.xlsx";
        input.onchange = (e) => {
          const file = (e.target as HTMLInputElement).files?.[0];
          if (file) {
            void handleFile(file);
          }
        };
        input.click();
      }}
      whileHover={{ scale: 1.005 }}
      whileTap={{ scale: 0.995 }}
    >
      <div className="flex items-start gap-3">
        {fileName ? (
          <>
            <FileSpreadsheet className="w-5 h-5 text-accent" />
            <div>
              <p className="text-sm font-medium text-foreground">{fileName}</p>
              <p className="text-[11px] text-accent font-mono">{isUploading ? "Uploading..." : "Dataset loaded"}</p>
            </div>
          </>
        ) : (
          <>
            {isUploading ? <Loader2 className="w-5 h-5 text-primary animate-spin" /> : <Upload className="w-5 h-5 text-muted-foreground" />}
            <div>
              <p className="text-sm text-foreground">Upload Dataset</p>
              <p className="text-[11px] text-muted-foreground">Drop a .csv, .json, or .xlsx file</p>
            </div>
          </>
        )}
      </div>
      {error && <p className="mt-3 text-[11px] text-destructive">{error}</p>}
    </motion.div>
  );
};

export default DatasetUpload;
