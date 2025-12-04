/**
 * API Client for DDQ Agent Backend
 * Handles all communication with FastAPI server
 */

const API_BASE = '';  // Same origin

class APIClient {
    constructor() {
        this.sessionId = null;
    }

    async request(endpoint, options = {}) {
        const url = `${API_BASE}${endpoint}`;
        const config = {
            headers: {
                'Content-Type': 'application/json',
            },
            ...options
        };

        try {
            const response = await fetch(url, config);
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || `HTTP ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error(`API Error [${endpoint}]:`, error);
            throw error;
        }
    }

    // ============== Model Endpoints ==============

    async listModels() {
        return this.request('/api/models');
    }

    async loadModel(modelType) {
        return this.request('/api/models/load', {
            method: 'POST',
            body: JSON.stringify({ model_type: modelType })
        });
    }

    async getCurrentModel() {
        return this.request('/api/models/current');
    }

    // ============== Conversation Endpoints ==============

    async startConversation(persona = 'random') {
        const response = await this.request('/api/conversation/start', {
            method: 'POST',
            body: JSON.stringify({ persona })
        });
        this.sessionId = response.session_id;
        return response;
    }

    async takeAction(actionId = null) {
        if (!this.sessionId) {
            throw new Error('No active session. Start a conversation first.');
        }
        return this.request('/api/conversation/action', {
            method: 'POST',
            body: JSON.stringify({
                session_id: this.sessionId,
                action: actionId
            })
        });
    }

    async autoPlay() {
        if (!this.sessionId) {
            throw new Error('No active session. Start a conversation first.');
        }
        return this.request('/api/conversation/auto-play', {
            method: 'POST',
            body: JSON.stringify({ session_id: this.sessionId })
        });
    }

    async getState() {
        if (!this.sessionId) {
            return null;
        }
        return this.request(`/api/conversation/state/${this.sessionId}`);
    }

    async endConversation() {
        if (!this.sessionId) {
            return;
        }
        await this.request(`/api/conversation/${this.sessionId}`, {
            method: 'DELETE'
        });
        this.sessionId = null;
    }

    // ============== Training Endpoints ==============

    async getTrainingHistory() {
        return this.request('/api/training/history');
    }

    async getHistoryFile(filename) {
        return this.request(`/api/training/history/${filename}`);
    }

    async listFigures() {
        return this.request('/api/training/figures');
    }

    // ============== Health ==============

    async healthCheck() {
        return this.request('/api/health');
    }

    async getSessions() {
        return this.request('/api/sessions');
    }
}

// Export singleton instance
const api = new APIClient();
