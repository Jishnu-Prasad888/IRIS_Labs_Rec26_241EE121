// App.tsx - Neo-Brutalist Redesign with Fixed Columns
import React, { useState, useEffect } from "react";
import {
  Container,
  Grid,
  Paper,
  Typography,
  TextField,
  Button,
  Box,
  CircularProgress,
  Chip,
  Card,
  CardContent,
  IconButton,
  Drawer,
  Divider,
  Slider,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  useMediaQuery,
  ThemeProvider,
  createTheme,
} from "@mui/material";
import {
  ContentCopy as ContentCopyIcon,
  Send as SendIcon,
  Refresh as RefreshIcon,
  Settings as SettingsIcon,
  Info as InfoIcon,
  ExpandMore as ExpandMoreIcon,
  Menu as MenuIcon,
  Close as CloseIcon,
} from "@mui/icons-material";
import { useTheme } from "@mui/material/styles";
import axios from "axios";
import ReactMarkdown from "react-markdown";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

// Neo-Brutalist Color System
const colors = {
  bg: "#F5F5F5",
  surface: "#FFFFFF",
  ink: "#111111",
  border: "#000000",
  primary: "#0057FF",
  secondary: "#FFB800",
  accent: "#FF3D00",
  success: "#00C853",
  error: "#D50000",
  warning: "#FF6D00",
  muted: "#E0E0E0",
};

// Neo-Brutalist Theme
const brutalistTheme = createTheme({
  palette: {
    background: {
      default: colors.bg,
      paper: colors.surface,
    },
    primary: {
      main: colors.primary,
      contrastText: "#FFFFFF",
    },
    secondary: {
      main: colors.secondary,
    },
    error: {
      main: colors.error,
    },
    text: {
      primary: colors.ink,
    },
  },
  shape: {
    borderRadius: 0,
  },
  typography: {
    fontFamily: `"Space Grotesk", "Inter", system-ui, sans-serif`,
    h1: {
      fontSize: "32px",
      fontWeight: 800,
    },
    h2: {
      fontSize: "24px",
      fontWeight: 700,
    },
    h3: {
      fontSize: "18px",
      fontWeight: 700,
    },
    body1: {
      fontSize: "14px",
      fontWeight: 500,
      lineHeight: 1.5,
    },
    caption: {
      fontSize: "12px",
      fontWeight: 600,
    },
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          border: "2px solid #000",
          boxShadow: "none",
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          border: "2px solid #000",
          boxShadow: "none",
          backgroundColor: colors.surface,
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          border: "2px solid #000",
          fontWeight: 700,
          textTransform: "none",
          boxShadow: "none",
          ":hover": {
            boxShadow: "4px 4px 0px #000",
            backgroundColor: "inherit",
          },
          "&.Mui-disabled": {
            borderColor: colors.muted,
            color: colors.muted,
          },
        },
        containedPrimary: {
          backgroundColor: colors.primary,
          color: "#FFFFFF",
          ":hover": {
            backgroundColor: colors.primary,
            boxShadow: "4px 4px 0px #000",
          },
        },
        outlined: {
          ":hover": {
            backgroundColor: colors.surface,
            boxShadow: "4px 4px 0px #000",
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          border: "2px solid #000",
          fontWeight: 600,
          borderRadius: "0px",
          "&:hover": {
            boxShadow: "2px 2px 0px #000",
          },
        },
        filledPrimary: {
          backgroundColor: colors.primary,
          color: "#FFFFFF",
        },
        filledSuccess: {
          backgroundColor: colors.success,
          color: "#FFFFFF",
        },
        filledError: {
          backgroundColor: colors.error,
          color: "#FFFFFF",
        },
        filledWarning: {
          backgroundColor: colors.warning,
          color: "#FFFFFF",
        },
      },
    },
    MuiAccordion: {
      styleOverrides: {
        root: {
          border: "2px solid #000",
          boxShadow: "none",
          "&:before": {
            display: "none",
          },
          "&.Mui-expanded": {
            margin: 0,
          },
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          "& .MuiOutlinedInput-root": {
            borderRadius: 0,
            "& fieldset": {
              border: "2px solid #000",
            },
            "&:hover fieldset": {
              border: "2px solid #000",
            },
            "&.Mui-focused fieldset": {
              border: "2px solid #000",
            },
          },
        },
      },
    },
    MuiSelect: {
      styleOverrides: {
        root: {
          borderRadius: 0,
          "& .MuiOutlinedInput-notchedOutline": {
            border: "2px solid #000",
          },
          "&:hover .MuiOutlinedInput-notchedOutline": {
            border: "2px solid #000",
          },
          "&.Mui-focused .MuiOutlinedInput-notchedOutline": {
            border: "2px solid #000",
          },
        },
      },
    },
    MuiSlider: {
      styleOverrides: {
        root: {
          color: colors.primary,
          "& .MuiSlider-thumb": {
            border: "2px solid #000",
            borderRadius: 0,
          },
          "& .MuiSlider-track": {
            border: "none",
          },
          "& .MuiSlider-rail": {
            backgroundColor: colors.muted,
          },
        },
      },
    },
    MuiDivider: {
      styleOverrides: {
        root: {
          borderColor: colors.border,
          borderWidth: "1px",
        },
      },
    },
  },
});

// Define types
interface SystemStatus {
  api_ready: boolean;
  system_ready: boolean;
  file_exists: boolean;
  file_size_kb: number;
  chunk_count: number;
}

interface SourceMetadata {
  chunk_type: string;
  level: number;
  similarity: number;
  has_parent: boolean;
  child_count: number;
  text_preview: string;
}

interface MessageMetadata {
  strategy: string;
  question_type: string;
  chunks_retrieved: number;
  sources: SourceMetadata[];
}

interface Message {
  id: number;
  role: "user" | "assistant" | "system" | "error";
  content: string;
  timestamp: string;
  metadata?: MessageMetadata;
}

interface Settings {
  kChunks: number;
  similarityThreshold: number;
  retrievalStrategy: string;
}

interface HierarchyStats {
  total_nodes: number;
  level_counts: Record<string, number>;
}

interface HierarchyData {
  stats: HierarchyStats;
  hierarchy: any;
}

function App() {
  const [copiedMessageId, setCopiedMessageId] = useState<number | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [initializing, setInitializing] = useState(false);
  const [settings, setSettings] = useState<Settings>({
    kChunks: 5,
    similarityThreshold: 0.25,
    retrievalStrategy: "adaptive",
  });
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [hierarchy, setHierarchy] = useState<HierarchyData | null>(null);
  const [mobileControlsOpen, setMobileControlsOpen] = useState(false);

  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down("md"));

  const copyToClipboard = async (text: string, messageId: number) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedMessageId(messageId);
      setTimeout(() => setCopiedMessageId(null), 2000); // Reset after 2 seconds
    } catch (err) {
      console.error("Failed to copy text: ", err);
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  const fetchStatus = async () => {
    try {
      const response = await axios.get<SystemStatus>(`${API_BASE_URL}/status`);
      setSystemStatus(response.data);
    } catch (error) {
      console.error("Error fetching status:", error);
    }
  };

  const initializeSystem = async () => {
    setInitializing(true);
    try {
      await axios.post(`${API_BASE_URL}/initialize`, {
        html_path: "data/raw/odyssey.html",
      });
      await fetchStatus();
      addMessage("system", "System initialized successfully!");
    } catch (error: any) {
      addMessage(
        "error",
        `Initialization failed: ${error.response?.data?.detail || error.message}`,
      );
    } finally {
      setInitializing(false);
    }
  };

  const askQuestion = async () => {
    if (!input.trim()) return;

    const userMessage = input;
    setInput("");
    addMessage("user", userMessage);
    setLoading(true);

    try {
      const response = await axios.post<{
        answer: string;
        strategy: string;
        question_type: string;
        chunks_retrieved: number;
        sources: SourceMetadata[];
      }>(`${API_BASE_URL}/ask`, {
        question: userMessage,
        k_chunks: settings.kChunks,
        similarity_threshold: settings.similarityThreshold,
        retrieval_strategy: settings.retrievalStrategy,
      });

      addMessage("assistant", response.data.answer, {
        strategy: response.data.strategy,
        question_type: response.data.question_type,
        chunks_retrieved: response.data.chunks_retrieved,
        sources: response.data.sources,
      });
    } catch (error: any) {
      addMessage(
        "error",
        `Error: ${error.response?.data?.detail || error.message}`,
      );
    } finally {
      setLoading(false);
    }
  };

  const addMessage = (
    role: "user" | "assistant" | "system" | "error",
    content: string,
    metadata?: MessageMetadata,
  ) => {
    const newMessage: Message = {
      id: Date.now(),
      role,
      content,
      timestamp: new Date().toLocaleTimeString(),
      metadata,
    };
    setMessages((prev) => [...prev, newMessage]);
  };

  const fetchHierarchy = async () => {
    try {
      const response = await axios.get<HierarchyData>(
        `${API_BASE_URL}/hierarchy`,
      );
      setHierarchy(response.data);
      setDrawerOpen(true);
    } catch (error) {
      console.error("Error fetching hierarchy:", error);
    }
  };

  const resetSystem = async () => {
    try {
      await axios.post(`${API_BASE_URL}/reset`);
      setMessages([]);
      fetchStatus();
      addMessage("system", "System reset successfully.");
    } catch (error) {
      console.error("Error resetting system:", error);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      askQuestion();
    }
  };

  // Neo-Brutalist Message Component
  const MessageBubble = ({ msg }: { msg: Message }) => {
    const isUser = msg.role === "user";
    const isError = msg.role === "error";
    const isSystem = msg.role === "system";
    const isCopied = copiedMessageId === msg.id;

    const bubbleStyles = {
      user: {
        bgcolor: colors.primary,
        color: "#FFFFFF",
        alignSelf: "flex-end",
        border: "2px solid #000",
      },
      assistant: {
        bgcolor: colors.surface,
        color: colors.ink,
        alignSelf: "flex-start",
        border: "2px solid #000",
      },
      error: {
        bgcolor: colors.error,
        color: "#FFFFFF",
        alignSelf: "flex-start",
        border: "2px solid #000",
      },
      system: {
        bgcolor: colors.muted,
        color: colors.ink,
        alignSelf: "center",
        border: "2px solid #000",
      },
    };

    const currentStyle = isUser
      ? bubbleStyles.user
      : isError
        ? bubbleStyles.error
        : isSystem
          ? bubbleStyles.system
          : bubbleStyles.assistant;

    // Show copy button for user and assistant messages only
    const showCopyButton = isUser || msg.role === "assistant";

    return (
      <Box
        sx={{
          display: "flex",
          flexDirection: "column",
          ...currentStyle,
          maxWidth: { xs: "85%", md: "75%" },
          p: 2,
          mb: 2,
          position: "relative",
        }}
      >
        {/* Header with Role, Timestamp, and Copy Button */}
        <Box
          sx={{
            display: "flex",
            gap: 1,
            mb: 1,
            alignItems: "center",
            flexWrap: "wrap",
          }}
        >
          <Typography
            variant="caption"
            sx={{
              bgcolor: isUser ? "#0046CC" : colors.muted,
              color: isUser ? "#FFFFFF" : colors.ink,
              px: 1,
              py: 0.5,
              border: "1px solid #000",
            }}
          >
            {msg.role.toUpperCase()}
          </Typography>
          <Typography
            variant="caption"
            sx={{
              bgcolor: colors.muted,
              color: colors.ink,
              px: 1,
              py: 0.5,
              border: "1px solid #000",
            }}
          >
            {msg.timestamp}
          </Typography>

          {/* Copy Button */}
          {showCopyButton && (
            <Box sx={{ marginLeft: "auto" }}>
              <IconButton
                onClick={() => copyToClipboard(msg.content, msg.id)}
                size="small"
                sx={{
                  border: "1px solid #000",
                  borderRadius: 0,
                  bgcolor: isCopied
                    ? colors.success
                    : isUser
                      ? "#0046CC"
                      : colors.surface,
                  color: isCopied ? "#FFFFFF" : isUser ? "#FFFFFF" : colors.ink,
                  width: 28,
                  height: 28,
                  "&:hover": {
                    bgcolor: isCopied
                      ? colors.success
                      : isUser
                        ? "#003399"
                        : colors.muted,
                    boxShadow: "2px 2px 0px #000",
                  },
                  transition: "all 0.2s",
                }}
              >
                {isCopied ? (
                  <Typography
                    variant="caption"
                    sx={{ fontSize: "10px", fontWeight: 700 }}
                  >
                    ✓
                  </Typography>
                ) : (
                  <ContentCopyIcon sx={{ fontSize: 14 }} />
                )}
              </IconButton>
            </Box>
          )}
        </Box>

        {/* Message Content */}
        <Typography variant="body1" sx={{ whiteSpace: "pre-wrap" }}>
          <ReactMarkdown>{msg.content}</ReactMarkdown>
        </Typography>

        {/* Retrieval Details (for assistant messages) */}
        {msg.metadata?.sources && (
          <Accordion sx={{ mt: 2, bgcolor: "transparent" }}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="caption">RETRIEVAL DETAILS</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1, mb: 2 }}>
                <Chip
                  label={`STRATEGY ${msg.metadata.strategy}`}
                  size="small"
                  color="primary"
                />
                <Chip
                  label={`TYPE ${msg.metadata.question_type}`}
                  size="small"
                  color="secondary"
                />
                <Chip
                  label={`CHUNKS ${msg.metadata.chunks_retrieved}`}
                  size="small"
                />
                <Chip
                  label={`AVG SIM ${
                    msg.metadata.sources.length > 0
                      ? (
                          msg.metadata.sources.reduce(
                            (acc: number, s) => acc + s.similarity,
                            0,
                          ) / msg.metadata.sources.length
                        ).toFixed(3)
                      : "N/A"
                  }`}
                  size="small"
                />
              </Box>
              <Typography variant="caption" sx={{ fontWeight: 700, mb: 1 }}>
                RETRIEVED SOURCES:
              </Typography>
              {msg.metadata.sources.map((source, index) => (
                <Accordion key={index} sx={{ mb: 1 }}>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="caption">
                      [{source.level}] {source.chunk_type}
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Box sx={{ display: "flex", gap: 1, mb: 1 }}>
                      <Chip
                        label={`SIMILARITY ${source.similarity.toFixed(3)}`}
                        size="small"
                      />
                      <Chip
                        label={`CHILDREN ${source.child_count}`}
                        size="small"
                      />
                    </Box>
                    <Typography variant="caption" sx={{ fontWeight: 700 }}>
                      PREVIEW
                    </Typography>
                    <Typography
                      variant="body2"
                      sx={{ fontSize: "11px", mt: 1 }}
                    >
                      {source.text_preview}
                    </Typography>
                  </AccordionDetails>
                </Accordion>
              ))}
            </AccordionDetails>
          </Accordion>
        )}
      </Box>
    );
  };

  // System Control Panel Component
  const SystemControlPanel = () => (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <Typography
        variant="h3"
        sx={{
          bgcolor: colors.ink,
          color: colors.surface,
          p: 1,
          textAlign: "center",
        }}
      >
        SYSTEM CONTROL
      </Typography>

      <Button
        variant="contained"
        onClick={initializeSystem}
        disabled={initializing || systemStatus?.system_ready}
        fullWidth
        sx={{ height: 48 }}
      >
        {initializing ? (
          <CircularProgress size={24} sx={{ color: "#FFFFFF" }} />
        ) : (
          "INITIALIZE SYSTEM"
        )}
      </Button>

      <Button
        variant="outlined"
        onClick={resetSystem}
        fullWidth
        sx={{ height: 48 }}
      >
        RESET SYSTEM
      </Button>

      <Button
        variant="outlined"
        onClick={fetchHierarchy}
        startIcon={<InfoIcon />}
        fullWidth
        sx={{ height: 48 }}
      >
        DOCUMENT STRUCTURE
      </Button>

      <Divider sx={{ my: 1 }} />

      <Typography
        variant="h3"
        sx={{
          bgcolor: colors.ink,
          color: colors.surface,
          p: 1,
          textAlign: "center",
        }}
      >
        RETRIEVAL SETTINGS
      </Typography>

      <Box>
        <Typography variant="caption" sx={{ fontWeight: 700 }}>
          MAX CHUNKS: {settings.kChunks}
        </Typography>
        <Slider
          value={settings.kChunks}
          onChange={(_, value) =>
            setSettings({ ...settings, kChunks: value as number })
          }
          min={1}
          max={10}
          step={1}
          marks
          valueLabelDisplay="auto"
        />
      </Box>

      <Box>
        <Typography variant="caption" sx={{ fontWeight: 700 }}>
          SIMILARITY THRESHOLD: {settings.similarityThreshold.toFixed(2)}
        </Typography>
        <Slider
          value={settings.similarityThreshold}
          onChange={(_, value) =>
            setSettings({
              ...settings,
              similarityThreshold: value as number,
            })
          }
          min={0}
          max={1}
          step={0.05}
          marks
          valueLabelDisplay="auto"
        />
      </Box>

      <FormControl fullWidth>
        <InputLabel>RETRIEVAL STRATEGY</InputLabel>
        <Select
          value={settings.retrievalStrategy}
          label="RETRIEVAL STRATEGY"
          onChange={(e) =>
            setSettings({ ...settings, retrievalStrategy: e.target.value })
          }
        >
          <MenuItem value="adaptive">ADAPTIVE</MenuItem>
          <MenuItem value="overview">OVERVIEW</MenuItem>
          <MenuItem value="detail">DETAIL</MenuItem>
          <MenuItem value="character">CHARACTER</MenuItem>
          <MenuItem value="structural">STRUCTURAL</MenuItem>
        </Select>
      </FormControl>
    </Box>
  );

  return (
    <ThemeProvider theme={brutalistTheme}>
      <Box
        sx={{
          bgcolor: colors.bg,
          minHeight: "100vh",
          display: "flex",
          flexDirection: "column",
        }}
      >
        {/* Header */}
        <Box
          sx={{
            bgcolor: colors.primary,
            color: "#FFFFFF",
            p: 2,
            borderBottom: "3px solid #000",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
            {isMobile && (
              <IconButton
                onClick={() => setMobileControlsOpen(!mobileControlsOpen)}
                sx={{
                  border: "2px solid #000",
                  borderRadius: 0,
                  bgcolor: colors.surface,
                }}
              >
                {mobileControlsOpen ? <CloseIcon /> : <MenuIcon />}
              </IconButton>
            )}
            <Typography variant="h1" sx={{ color: "#FFFFFF" }}>
              THE ODYSSEY
            </Typography>
          </Box>

          {!isMobile && (
            <Button
              variant="outlined"
              onClick={() => setDrawerOpen(true)}
              startIcon={<InfoIcon />}
              sx={{
                border: "2px solid #FFFFFF",
                color: "#FFFFFF",
                "&:hover": { boxShadow: "4px 4px 0px #000" },
              }}
            >
              HIERARCHY
            </Button>
          )}
        </Box>

        {/* Status Bar */}
        <Box
          sx={{
            bgcolor: colors.surface,
            borderBottom: "2px solid #000",
            p: 1.5,
            display: "flex",
            gap: 1,
            flexWrap: "wrap",
            justifyContent: "center",
          }}
        >
          {[
            {
              label: "API STATUS",
              status: systemStatus?.api_ready,
              value: systemStatus?.api_ready ? "READY" : "MISSING",
              color: systemStatus?.api_ready ? "success" : "error",
            },
            {
              label: "SYSTEM STATUS",
              status: systemStatus?.system_ready,
              value: systemStatus?.system_ready ? "READY" : "NOT READY",
              color: systemStatus?.system_ready ? "success" : "warning",
              sub: systemStatus?.chunk_count
                ? `${systemStatus.chunk_count} chunks`
                : undefined,
            },
            {
              label: "SOURCE FILE",
              status: systemStatus?.file_exists,
              value: systemStatus?.file_exists ? "FOUND" : "MISSING",
              color: systemStatus?.file_exists ? "success" : "error",
              sub: systemStatus?.file_size_kb
                ? `${systemStatus.file_size_kb.toFixed(1)} KB`
                : undefined,
            },
            {
              label: "MESSAGES",
              value: messages.length.toString(),
              color: "default",
            },
          ].map((indicator, index) => (
            <Box
              key={index}
              sx={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
              }}
            >
              <Chip
                label={indicator.value}
                color={indicator.color as any}
                size="small"
                sx={{ mb: 0.5 }}
              />
              <Typography variant="caption" sx={{ fontSize: "10px" }}>
                {indicator.label}
              </Typography>
              {indicator.sub && (
                <Typography
                  variant="caption"
                  sx={{ fontSize: "9px", color: colors.muted }}
                >
                  {indicator.sub}
                </Typography>
              )}
            </Box>
          ))}
        </Box>

        {/* Main Content Area - FIXED SPLIT */}
        <Box
          sx={{
            display: "flex",
            flexGrow: 1,
            overflow: "hidden",
            height: "calc(100vh - 140px)", // Subtract header + status bar height
          }}
        >
          {/* LEFT COLUMN - Chat (75% on desktop, 100% on mobile) */}
          <Box
            sx={{
              width: { xs: "100%", md: "75%" },
              display: "flex",
              flexDirection: "column",
              borderRight: { xs: "none", md: "3px solid #000" },
              overflow: "hidden",
            }}
          >
            {/* Messages Area - Scrollable */}
            <Box
              sx={{
                flexGrow: 1,
                overflowY: "auto",
                p: 2,
                display: "flex",
                flexDirection: "column",
              }}
            >
              {messages.map((msg) => (
                <MessageBubble key={msg.id} msg={msg} />
              ))}

              {loading && (
                <Box
                  sx={{
                    display: "flex",
                    alignItems: "center",
                    gap: 2,
                    p: 2,
                    border: "2px solid #000",
                    bgcolor: colors.muted,
                    alignSelf: "flex-start",
                  }}
                >
                  <Typography variant="caption" sx={{ fontWeight: 700 }}>
                    ASSISTANT IS THINKING
                  </Typography>
                  {[0, 1, 2].map((i) => (
                    <Box
                      key={i}
                      sx={{
                        width: 8,
                        height: 8,
                        bgcolor: colors.primary,
                        animation: `pulse 1.5s ease-in-out ${i * 0.2}s infinite`,
                        "@keyframes pulse": {
                          "0%, 100%": { opacity: 0.3 },
                          "50%": { opacity: 1 },
                        },
                      }}
                    />
                  ))}
                </Box>
              )}
            </Box>

            {/* Input Area - Fixed at Bottom */}
            <Box
              sx={{
                p: 2,
                bgcolor: colors.surface,
                borderTop: "2px solid #000",
              }}
            >
              {/* Example Questions */}
              <Box sx={{ mb: 2 }}>
                <Typography
                  variant="caption"
                  sx={{ fontWeight: 700, mb: 1, display: "block" }}
                >
                  QUICK QUESTIONS
                </Typography>
                <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
                  {[
                    "Tell me about the Cyclops",
                    "What is the role of Athena?",
                    "Summary of Book 1",
                    "Who is Telemachus?",
                    "What happens in the underworld?",
                  ].map((question) => (
                    <Chip
                      key={question}
                      label={question}
                      onClick={() => setInput(question)}
                      size="small"
                      variant="outlined"
                      sx={{
                        cursor: "pointer",
                        fontSize: "11px",
                        height: "24px",
                        "& .MuiChip-label": { px: 1 },
                      }}
                    />
                  ))}
                </Box>
              </Box>

              {/* Input Field */}
              <Box sx={{ display: "flex", gap: 2 }}>
                <TextField
                  fullWidth
                  multiline
                  maxRows={4}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="ASK ABOUT THE ODYSSEY..."
                  disabled={!systemStatus?.system_ready}
                  variant="outlined"
                  sx={{
                    "& .MuiOutlinedInput-root": {
                      fontSize: "14px",
                      fontWeight: 500,
                    },
                  }}
                />
                <Button
                  variant="contained"
                  onClick={askQuestion}
                  disabled={!systemStatus?.system_ready || !input.trim()}
                  endIcon={<SendIcon />}
                  sx={{ minWidth: 100, height: 56 }}
                >
                  ASK
                </Button>
              </Box>
            </Box>
          </Box>

          {/* RIGHT COLUMN - System Controls (25% on desktop, drawer on mobile) */}
          {!isMobile && (
            <Box
              sx={{
                width: "25%",
                overflowY: "auto",
                p: 2,
                bgcolor: colors.surface,
              }}
            >
              <SystemControlPanel />
            </Box>
          )}
        </Box>

        {/* Mobile Controls Drawer */}
        <Drawer
          anchor="left"
          open={mobileControlsOpen}
          onClose={() => setMobileControlsOpen(false)}
          sx={{
            "& .MuiDrawer-paper": {
              borderRight: "3px solid #000",
              width: "85%",
              maxWidth: 300,
              bgcolor: colors.surface,
            },
          }}
        >
          <Box sx={{ p: 2 }}>
            <Box
              sx={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                mb: 2,
              }}
            >
              <Typography variant="h3">CONTROLS</Typography>
              <IconButton onClick={() => setMobileControlsOpen(false)}>
                <CloseIcon />
              </IconButton>
            </Box>
            <SystemControlPanel />
          </Box>
        </Drawer>

        {/* Hierarchy Drawer */}
        <Drawer
          anchor="right"
          open={drawerOpen}
          onClose={() => setDrawerOpen(false)}
          sx={{
            "& .MuiDrawer-paper": {
              borderLeft: "3px solid #000",
              width: { xs: "100%", sm: 400 },
              bgcolor: colors.surface,
            },
          }}
        >
          <Box sx={{ p: 2 }}>
            <Box
              sx={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                mb: 2,
              }}
            >
              <Typography variant="h3">DOCUMENT HIERARCHY</Typography>
              <IconButton onClick={() => setDrawerOpen(false)}>
                <CloseIcon />
              </IconButton>
            </Box>

            {hierarchy ? (
              <>
                <Typography
                  variant="caption"
                  sx={{
                    fontWeight: 700,
                    display: "block",
                    mb: 1,
                    bgcolor: colors.muted,
                    p: 1,
                  }}
                >
                  STATISTICS
                </Typography>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    TOTAL NODES: {hierarchy.stats.total_nodes}
                  </Typography>
                  {Object.entries(hierarchy.stats.level_counts).map(
                    ([level, count]) => (
                      <Typography key={level} variant="body2">
                        LEVEL {level}: <strong>{count as number} nodes</strong>
                      </Typography>
                    ),
                  )}
                </Box>
                <Typography variant="body2" sx={{ mb: 2 }}>
                  Hierarchy structure loaded from hierarchy.json
                </Typography>
                <Button
                  variant="outlined"
                  onClick={() => {
                    console.log(hierarchy.hierarchy);
                    alert("Full hierarchy data logged to console");
                  }}
                  sx={{ mt: 2, height: 48 }}
                  fullWidth
                >
                  VIEW FULL JSON IN CONSOLE
                </Button>
              </>
            ) : (
              <Box sx={{ textAlign: "center", py: 4 }}>
                <CircularProgress />
                <Typography variant="body2" sx={{ mt: 2 }}>
                  LOADING HIERARCHY...
                </Typography>
              </Box>
            )}
          </Box>
        </Drawer>
      </Box>
    </ThemeProvider>
  );
}

export default App;
