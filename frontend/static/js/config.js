// config.js
export const config = {
    SERVER: {
        HOST: 'localhost',
        PORT: 3000,
        get URL() {
            return `http://${this.HOST}:${this.PORT}`;
        }
    },

    main_server: {
        port: 3000,
        host: 'localhost',
        protocol: 'http',
        get URL() {
            return `${this.protocol}://${this.host}:${this.port}`;
        }
    },

    // دالة للتحقق من الهوست والحصول على IP المحلي
    async checkHostAndGetLocalIP(host = 'localhost') {
        try {
            const response = await fetch(`http://${host}:${this.SERVER.PORT}/api/network/local-ip`, {
                method: 'GET',
                headers: { 'Cache-Control': 'no-cache' },
                timeout: 2000
            });

            if (!response.ok) {
                throw new Error('الهوست غير متاح');
            }

            const data = await response.json();
            return {
                isAvailable: true,
                localIP: data.success ? data.localIP : 'localhost'
            };
        } catch (error) {
            console.warn(`خطأ في الاتصال بالهوست ${host}:`, error);
            return {
                isAvailable: false,
                localIP: 'localhost'
            };
        }
    },

    // دالة لتحديث الهوست في جميع الإعدادات
    _updateHostInSettings(host) {
        this.SERVER.HOST = host;
        this.main_server.host = host;
        localStorage.setItem('server_host', host);
    },

    // دالة للحصول على الهوست الحالي
    getCurrentHost() {
        const savedHost = localStorage.getItem('server_host');
        if (savedHost) {
            this._updateHostInSettings(savedHost);
            return `http://${savedHost}:${this.SERVER.PORT}`;
        }
        return this.SERVER.URL;
    },

    // دالة للحصول على عنوان API الرئيسي
    getApiMainServerUrl() {
        const savedHost = localStorage.getItem('server_host');
        if (savedHost) {
            this.main_server.host = savedHost;
            return `${this.main_server.protocol}://${savedHost}:${this.main_server.port}`;
        }
        return this.main_server.URL;
    },

    // دالة لتحديث الهوست
    async updateHost(newHost) {
        try {
            // إذا كان الهوست هو نفسه الحالي، لا داعي للتحقق
            if (newHost === this.SERVER.HOST) {
                return true;
            }

            // التحقق من الهوست والحصول على IP المحلي
            const { isAvailable, localIP } = await this.checkHostAndGetLocalIP(newHost);
            
            // تحديث الهوست المناسب
            this._updateHostInSettings(isAvailable ? newHost : localIP);
            
            return isAvailable;
        } catch (error) {
            console.error('خطأ في تحديث الهوست:', error);
            return false;
        }
    },

    // دالة التهيئة
    async initialize() {
        try {
            // التحقق من الهوست الحالي والحصول على IP المحلي
            const { localIP } = await this.checkHostAndGetLocalIP();
            const savedHost = localStorage.getItem('server_host') || localIP;
            
            // تحديث الهوست
            await this.updateHost(savedHost);
        } catch (error) {
            console.error('خطأ في تهيئة التكوين:', error);
            this._updateHostInSettings('localhost');
        }
    }
};

// تهيئة التكوين عند تحميل الملف
document.addEventListener('DOMContentLoaded', () => {
    config.initialize();
});
