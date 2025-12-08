"use client";

import { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { ChatPanel } from "@/components/chat-panel";
import { QValueChart } from "@/components/q-value-chart";
import { api, EnvironmentState, ActionResponse } from "@/lib/api";

const ACTION_NAMES = [
  "Empathetic Listening",
  "Ask About Situation",
  "Firm Reminder",
  "Offer Payment Plan",
  "Propose Settlement",
  "Hard Close",
  "Schedule Callback",
  "Escalate",
  "End Call",
];

interface Message {
  role: "agent" | "debtor" | "system";
  content: string;
  action?: string;
}

export default function DemoPage() {
  const [modelType, setModelType] = useState("ddq");
  const [difficulty, setDifficulty] = useState("random");
  const [isLoading, setIsLoading] = useState(false);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [isConversationActive, setIsConversationActive] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [qValues, setQValues] = useState<number[]>([]);
  const [selectedAction, setSelectedAction] = useState<number | undefined>();
  const [state, setState] = useState<EnvironmentState | null>(null);
  const [status, setStatus] = useState("Ready to start");
  const [totalReward, setTotalReward] = useState(0);
  const [manualAction, setManualAction] = useState("0");

  // Check health and current model on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        await api.healthCheck();
        const currentModel = await api.getCurrentModel();
        if (currentModel.loaded) {
          setIsModelLoaded(true);
          setModelType(currentModel.model_type || "ddq");
          setStatus(`${currentModel.model_type?.toUpperCase()} model loaded`);
        }
      } catch (error) {
        console.error("Health check failed:", error);
        setStatus("Backend unavailable - start FastAPI server");
      }
    };
    checkHealth();
  }, []);

  const loadModel = useCallback(async () => {
    setIsLoading(true);
    setStatus(`Loading ${modelType.toUpperCase()} model...`);
    try {
      await api.loadModel(modelType);
      setIsModelLoaded(true);
      setStatus(`${modelType.toUpperCase()} model loaded successfully`);
    } catch (error) {
      setStatus(`Failed to load model: ${error}`);
    } finally {
      setIsLoading(false);
    }
  }, [modelType]);

  const startConversation = useCallback(async () => {
    setIsLoading(true);
    setStatus("Starting conversation...");
    try {
      const response = await api.startConversation(difficulty);
      setIsConversationActive(true);
      setState(response.state);
      setMessages([
        {
          role: "system",
          content: `Debtor: ${response.debtor_profile.name} | Debt: ₹${response.debtor_profile.debt_amount} | Due: ${response.debtor_profile.due_days} days | Personality: ${response.debtor_profile.personality}`,
        },
        {
          role: "debtor",
          content: response.initial_message,
        },
      ]);
      setQValues([]);
      setTotalReward(0);
      setStatus("Conversation started - select action or use agent");
    } catch (error) {
      setStatus(`Failed to start: ${error}`);
    } finally {
      setIsLoading(false);
    }
  }, [difficulty]);

  const takeAction = useCallback(async (actionId: number | null = null) => {
    setIsLoading(true);
    try {
      const response: ActionResponse = await api.takeAction(actionId);

      setMessages((prev) => [
        ...prev,
        {
          role: "agent",
          content: response.agent_response,
          action: ACTION_NAMES[response.action],
        },
        ...(response.debtor_response
          ? [{ role: "debtor" as const, content: response.debtor_response }]
          : []),
      ]);

      setQValues(response.q_values);
      setSelectedAction(response.action);
      setState(response.state);
      setTotalReward((prev) => prev + response.reward);

      if (response.done) {
        setIsConversationActive(false);
        setMessages((prev) => [
          ...prev,
          {
            role: "system",
            content: `Conversation ended: ${response.info?.outcome || "Complete"} | Total Reward: ${totalReward + response.reward}`,
          },
        ]);
        setStatus(`Ended: ${response.info?.outcome}`);
      } else {
        setStatus(`Action: ${ACTION_NAMES[response.action]} | Reward: ${response.reward.toFixed(2)}`);
      }
    } catch (error) {
      setStatus(`Error: ${error}`);
    } finally {
      setIsLoading(false);
    }
  }, [totalReward]);

  const autoPlay = useCallback(async () => {
    setIsLoading(true);
    setStatus("Running auto-play...");
    try {
      const response = await api.autoPlay();

      const newMessages: Message[] = [];
      response.steps.forEach((step) => {
        newMessages.push({
          role: "agent",
          content: step.agent_response,
          action: ACTION_NAMES[step.action],
        });
        if (step.debtor_response) {
          newMessages.push({
            role: "debtor",
            content: step.debtor_response,
          });
        }
        setQValues(step.q_values);
        setSelectedAction(step.action);
        setState(step.state);
      });

      newMessages.push({
        role: "system",
        content: `Conversation ended: ${response.final_outcome} | Total Reward: ${response.total_reward.toFixed(2)} | Turns: ${response.turns}`,
      });

      setMessages((prev) => [...prev, ...newMessages]);
      setTotalReward(response.total_reward);
      setIsConversationActive(false);
      setStatus(`Auto-play complete: ${response.final_outcome}`);
    } catch (error) {
      setStatus(`Auto-play error: ${error}`);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const endConversation = useCallback(async () => {
    await api.endConversation();
    setIsConversationActive(false);
    setMessages([]);
    setQValues([]);
    setState(null);
    setTotalReward(0);
    setStatus("Conversation ended");
  }, []);

  return (
    <div className="space-y-6">
      {/* Control Panel */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg">Control Panel</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">Model:</span>
              <Select value={modelType} onValueChange={setModelType}>
                <SelectTrigger className="w-[100px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="ddq">DDQ</SelectItem>
                  <SelectItem value="dqn">DQN</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">Difficulty:</span>
              <Select value={difficulty} onValueChange={setDifficulty}>
                <SelectTrigger className="w-[100px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="random">Random</SelectItem>
                  <SelectItem value="easy">Easy</SelectItem>
                  <SelectItem value="medium">Medium</SelectItem>
                  <SelectItem value="hard">Hard</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <Button
              variant="outline"
              onClick={loadModel}
              disabled={isLoading}
            >
              Load Model
            </Button>

            <Button
              variant="default"
              onClick={isConversationActive ? endConversation : startConversation}
              disabled={isLoading || !isModelLoaded}
            >
              {isConversationActive ? "End" : "Start"}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Status Bar */}
      <div className="flex items-center gap-2 px-4 py-2 bg-muted rounded-lg">
        <Badge variant={isModelLoaded ? "default" : "secondary"}>
          {isModelLoaded ? "Model Ready" : "No Model"}
        </Badge>
        <span className="text-sm">{status}</span>
        {isLoading && <span className="ml-auto text-sm text-muted-foreground">Loading...</span>}
      </div>

      {/* Main Content */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Left: Conversation */}
        <div className="space-y-4">
          <ChatPanel messages={messages} height="h-[350px]" />

          {isConversationActive && (
            <Card>
              <CardContent className="pt-4">
                <div className="flex flex-wrap items-center gap-2">
                  <Select value={manualAction} onValueChange={setManualAction}>
                    <SelectTrigger className="w-[180px]">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {ACTION_NAMES.map((name, idx) => (
                        <SelectItem key={idx} value={idx.toString()}>
                          {name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Button
                    variant="outline"
                    onClick={() => takeAction(parseInt(manualAction))}
                    disabled={isLoading}
                  >
                    Send Manual
                  </Button>
                  <Button
                    onClick={() => takeAction(null)}
                    disabled={isLoading}
                  >
                    Agent Action
                  </Button>
                  <Button
                    variant="secondary"
                    onClick={autoPlay}
                    disabled={isLoading}
                  >
                    Auto-Play
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Right: State & Q-Values */}
        <div className="space-y-4">
          {/* State Panel */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Environment State</CardTitle>
            </CardHeader>
            <CardContent>
              {state ? (
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-xs text-muted-foreground mb-1">Trust Level</div>
                    <Progress value={state.trust_level * 100} className="h-2" />
                    <div className="text-xs text-right">{(state.trust_level * 100).toFixed(0)}%</div>
                  </div>
                  <div>
                    <div className="text-xs text-muted-foreground mb-1">Anger Level</div>
                    <Progress value={state.anger_level * 100} className="h-2" />
                    <div className="text-xs text-right">{(state.anger_level * 100).toFixed(0)}%</div>
                  </div>
                  <div>
                    <div className="text-xs text-muted-foreground mb-1">Payment Likelihood</div>
                    <Progress value={state.payment_likelihood * 100} className="h-2" />
                    <div className="text-xs text-right">{(state.payment_likelihood * 100).toFixed(0)}%</div>
                  </div>
                  <div>
                    <div className="text-xs text-muted-foreground mb-1">Willingness to Pay</div>
                    <Progress value={state.willingness_to_pay * 100} className="h-2" />
                    <div className="text-xs text-right">{(state.willingness_to_pay * 100).toFixed(0)}%</div>
                  </div>
                  <div className="col-span-2 flex justify-between text-sm pt-2 border-t">
                    <span>Total Reward: <strong>{totalReward.toFixed(2)}</strong></span>
                    <span>Turns: <strong>{state.conversation_length}</strong></span>
                  </div>
                </div>
              ) : (
                <p className="text-center text-muted-foreground py-4 text-sm">
                  Start a conversation to see state
                </p>
              )}
            </CardContent>
          </Card>

          {/* Q-Value Chart */}
          <QValueChart qValues={qValues} selectedAction={selectedAction} />
        </div>
      </div>
    </div>
  );
}
