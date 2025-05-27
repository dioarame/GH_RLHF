// static/js/utils/APIClient.js

export class APIClient {
    static async fetchJSON(url, options = {}) {
        try {
            const response = await fetch(url, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error(`API 오류: ${url}`, error);
            throw error;
        }
    }
    
    static async getNextComparison() {
        return this.fetchJSON('/api/comparison/next', {
            method: 'POST',
            body: JSON.stringify({ session_id: 'rlhf_session' })
        });
    }
    
    static async submitFeedback(feedbackData) {
        return this.fetchJSON('/api/feedback/comparison', {
            method: 'POST',
            body: JSON.stringify(feedbackData)
        });
    }
    
    static async getDesignStats() {
        return this.fetchJSON('/api/designs/stats');
    }
    
    static async getMeshData(designId) {
        return this.fetchJSON(`/api/mesh/${designId}`);
    }
    
    static async loadEnvironmentData(filename) {
        const response = await fetch(`/data/environment/${filename}`);
        if (response.ok) {
            return await response.json();
        }
        return null;
    }
}