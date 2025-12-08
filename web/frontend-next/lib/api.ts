/**
 * API Client for DDQ Agent Backend
 * Typed TypeScript version for Next.js
 */

const API_BASE = '';

// ============== Types ==============

export interface Model {
    name: string;
    type: string;
    path: string;
}

export interface ModelsResponse {
    models: Model[];
}

export interface LoadModelResponse {
    status: string;
    model_type: string;
    message: string;
}

export interface CurrentModelResponse {
    model_type: string | null;
    loaded: boolean;
}

export interface ConversationStart {
    session_id: string;
    debtor_profile: {
        name: string;
        debt_amount: number;
        due_days: number;
        personality: string;
        preferred_language: string;
    };
    initial_message: string;
    state: EnvironmentState;
}

export interface EnvironmentState {
    trust_level: number;
    anger_level: number;
    payment_likelihood: number;
    conversation_length: number;
    willingness_to_pay: number;
    financial_capability: number;
    debt_amount: number;
    due_days: number;
}

export interface ActionResponse {
    action: number;
    action_name: string;
    agent_response: string;
    debtor_response: string;
    q_values: number[];
    reward: number;
    done: boolean;
    state: EnvironmentState;
    info?: {
        outcome?: string;
        payment_collected?: number;
        turns?: number;
    };
}

export interface AutoPlayResponse {
    steps: ActionResponse[];
    final_outcome: string;
    total_reward: number;
    turns: number;
}

export interface HealthResponse {
    status: string;
    llm_available: boolean;
    models_loaded: {
        dqn: boolean;
        ddq: boolean;
    };
}

export interface TrainingHistoryResponse {
    files: string[];
}

export interface SessionsResponse {
    active_count: number;
    sessions: Record<string, unknown>[];
}

// ============== API Client ==============

class APIClient {
    private sessionId: string | null = null;

    private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
        const url = `${API_BASE}${endpoint}`;
        const config: RequestInit = {
            headers: {
                'Content-Type': 'application/json',
            },
            ...options,
        };

        const response = await fetch(url, config);

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        return response.json();
    }

    // ============== Model Endpoints ==============

    async listModels(): Promise<ModelsResponse> {
        return this.request('/api/models');
    }

    async loadModel(modelType: string): Promise<LoadModelResponse> {
        return this.request('/api/models/load', {
            method: 'POST',
            body: JSON.stringify({ model_type: modelType }),
        });
    }

    async getCurrentModel(): Promise<CurrentModelResponse> {
        return this.request('/api/models/current');
    }

    // ============== Conversation Endpoints ==============

    async startConversation(persona: string = 'random'): Promise<ConversationStart> {
        const response = await this.request<ConversationStart>('/api/conversation/start', {
            method: 'POST',
            body: JSON.stringify({ persona }),
        });
        this.sessionId = response.session_id;
        return response;
    }

    async takeAction(actionId: number | null = null): Promise<ActionResponse> {
        if (!this.sessionId) {
            throw new Error('No active session. Start a conversation first.');
        }
        return this.request('/api/conversation/action', {
            method: 'POST',
            body: JSON.stringify({
                session_id: this.sessionId,
                action: actionId,
            }),
        });
    }

    async autoPlay(): Promise<AutoPlayResponse> {
        if (!this.sessionId) {
            throw new Error('No active session. Start a conversation first.');
        }
        return this.request('/api/conversation/auto-play', {
            method: 'POST',
            body: JSON.stringify({ session_id: this.sessionId }),
        });
    }

    async getState(): Promise<EnvironmentState | null> {
        if (!this.sessionId) {
            return null;
        }
        return this.request(`/api/conversation/state/${this.sessionId}`);
    }

    async endConversation(): Promise<void> {
        if (!this.sessionId) {
            return;
        }
        await this.request(`/api/conversation/${this.sessionId}`, {
            method: 'DELETE',
        });
        this.sessionId = null;
    }

    getSessionId(): string | null {
        return this.sessionId;
    }

    // ============== Training Endpoints ==============

    async getTrainingHistory(): Promise<TrainingHistoryResponse> {
        return this.request('/api/training/history');
    }

    async getHistoryFile(filename: string): Promise<unknown> {
        return this.request(`/api/training/history/${filename}`);
    }

    async listFigures(): Promise<{ figures: string[] }> {
        return this.request('/api/training/figures');
    }

    // ============== Health ==============

    async healthCheck(): Promise<HealthResponse> {
        return this.request('/api/health');
    }

    async getSessions(): Promise<SessionsResponse> {
        return this.request('/api/sessions');
    }
}

// Export singleton instance
export const api = new APIClient();
