// المتغيرات العامة
let currentRequestId = null;
let checkStatusInterval = null;
let emailCount = 1;
const maxEmails = 10;

// دالة لإضافة حقل بريد إلكتروني جديد
function addEmailField() {
    if (emailCount >= maxEmails) {
        showMessage('error', 'لا يمكن إضافة المزيد من عناوين البريد الإلكتروني');
        return;
    }
    
    const emailList = document.getElementById('additionalEmails');
    const emailItem = document.createElement('div');
    emailItem.className = 'email-item';
    emailItem.innerHTML = `
        <input type="email" class="form-control additional-email" placeholder="بريد إلكتروني إضافي">
        <button type="button" class="btn btn-danger" onclick="removeEmailField(this)">
            <i class="fas fa-times"></i>
        </button>
    `;
    emailList.appendChild(emailItem);
    emailCount++;
}

// دالة لحذف حقل بريد إلكتروني
function removeEmailField(button) {
    button.parentElement.remove();
    emailCount--;
}


// دالة لعرض قسم الانتظار
function showWaitingSection() {
    const formSection = document.querySelector('.license-form');
    const waitingSection = document.createElement('div');
    waitingSection.className = 'waiting-section';
    waitingSection.innerHTML = `
        <div class="status-icon waiting">
            <i class="fas fa-spinner fa-3x fa-spin"></i>
        </div>
        <h3>جاري معالجة طلبك</h3>
        <p>سيتم تحميل الترخيص تلقائياً بمجرد الموافقة على طلبك</p>
        <div class="progress">
            <div class="progress-bar"></div>
        </div>
        <button class="btn btn-secondary cancel-request-btn" onclick="cancelRequest()">
            إلغاء الطلب
        </button>
    `;
    formSection.replaceWith(waitingSection);
}

// دالة للتحقق من حالة الطلب
async function checkRequestStatus() {
    if (!currentRequestId) return;
    
    try {
        const response = await fetch(`/check-request-status/${currentRequestId}`);
        const data = await response.json();
        
        if (response.ok) {
            if (data.status === 'approved') {
                stopStatusCheck();
                showDownloadSection(data.license_id);
            } else if (data.status === 'rejected') {
                stopStatusCheck();
                showMessage('error', 'تم رفض طلبك');
                resetForm();
            }
        }
    } catch (error) {
        console.error('خطأ في التحقق من حالة الطلب:', error);
    }
}

// دالة لبدء التحقق الدوري من حالة الطلب
function startStatusCheck() {
    if (checkStatusInterval) return;
    checkStatusInterval = setInterval(checkRequestStatus, 10000); // كل 10 ثواني
}

// دالة لإيقاف التحقق الدوري
function stopStatusCheck() {
    if (checkStatusInterval) {
        clearInterval(checkStatusInterval);
        checkStatusInterval = null;
    }
}

// دالة لعرض قسم التحميل
function showDownloadSection(licenseId) {
    const downloadSection = document.querySelector('.download-section');
    if (downloadSection) {
        downloadSection.innerHTML = `
            <div class="status-icon approved">
                <i class="fas fa-check-circle fa-3x"></i>
            </div>
            <h3>تمت الموافقة على طلبك!</h3>
            <div class="download-container">
                <button class="download-btn" onclick="downloadLicense('${licenseId}')">
                    تحميل الترخيص <i class="fas fa-download"></i>
                </button>
                <button class="cancel-btn" onclick="cancelRequest()">
                    إلغاء الطلب <i class="fas fa-times"></i>
                </button>
            </div>
        `;
    }
}

// دالة لتحميل الترخيص
async function downloadLicense(licenseId) {
    try {
        const response = await fetch(`/download-license/${licenseId}`);
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'license.key';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            // إغلاق النافذة بعد التحميل
            setTimeout(() => {
                window.close();
            }, 1000);
        } else {
            showMessage('error', 'حدث خطأ في تحميل الترخيص');
        }
    } catch (error) {
        showMessage('error', 'حدث خطأ في الاتصال بالخادم');
    }
}

// دالة لإلغاء الطلب
async function cancelRequest() {
    if (!currentRequestId) return;
    
    try {
        const response = await fetch(`/cancel-request/${currentRequestId}`, {
            method: 'POST'
        });
        
        if (response.ok) {
            stopStatusCheck();
            resetForm();
            showMessage('success', 'تم إلغاء الطلب بنجاح');
        } else {
            showMessage('error', 'حدث خطأ في إلغاء الطلب');
        }
    } catch (error) {
        showMessage('error', 'حدث خطأ في الاتصال بالخادم');
    }
}


// دالة لعرض الرسائل
function showMessage(type, message) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type === 'error' ? 'danger' : 'success'}`;
    alertDiv.textContent = message;
    
    const container = document.querySelector('.modal-content');
    container.insertBefore(alertDiv, container.firstChild);
    
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}


// تحديث معالج النموذج
document.getElementById('licenseRequestForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const mainEmail = document.getElementById('mainEmail').value;
    const additionalEmails = Array.from(document.querySelectorAll('#additionalEmails input'))
        .map(input => input.value)
        .filter(email => email);
    const subscriptionPeriod = document.querySelector('input[name="subscriptionPeriod"]:checked').value;

    try {
        const response = await fetch('/request-license', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                main_email: mainEmail,
                additional_emails: additionalEmails,
                subscription_period: subscriptionPeriod
            })
        });

        const data = await response.json();
        
        if (response.ok) {
            // إخفاء زر تقديم الطلب وإظهار أزرار التحميل والإلغاء
            document.getElementById('submitRequestBtn').style.display = 'none';
            document.getElementById('downloadLicenseBtn').style.display = 'inline-block';
            document.getElementById('cancelRequestBtn').style.display = 'inline-block';
            
            // تخزين معرف الطلب
            localStorage.setItem('licenseRequestId', data.request_id);
            
            document.getElementById('successMessage').textContent = 'تم استلام طلبك بنجاح. يمكنك الآن تحميل الترخيص بعد الموافقة عليه.';
            document.getElementById('successMessage').style.display = 'block';
            document.getElementById('errorMessage').style.display = 'none';
        } else {
            throw new Error(data.message || 'حدث خطأ في معالجة الطلب');
        }
    } catch (error) {
        document.getElementById('errorMessage').textContent = error.message;
        document.getElementById('errorMessage').style.display = 'block';
        document.getElementById('successMessage').style.display = 'none';
    }
});

// دالة تحميل الترخيص
async function downloadLicense() {
    try {
        const requestId = localStorage.getItem('licenseRequestId');
        if (!requestId) {
            throw new Error('لم يتم العثور على معرف الطلب');
        }

        const hardwareId = await generateHardwareId();
        
        const response = await fetch('/generate-license', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                request_id: requestId,
                hardware_id: hardwareId
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'فشل في تحميل الترخيص');
        }

        // تحميل الملف
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = 'license.enc';
        document.body.appendChild(a);
        a.click();
        
        // تنظيف
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

        // عرض رسالة نجاح
        document.getElementById('successMessage').textContent = 'تم تحميل الترخيص بنجاح';
        document.getElementById('successMessage').style.display = 'block';
        document.getElementById('errorMessage').style.display = 'none';

    } catch (error) {
        document.getElementById('errorMessage').textContent = error.message;
        document.getElementById('errorMessage').style.display = 'block';
        document.getElementById('successMessage').style.display = 'none';
    }
}

// دالة إلغاء الطلب
async function cancelRequest() {
    try {
        const requestId = localStorage.getItem('licenseRequestId');
        if (!requestId) {
            throw new Error('لم يتم العثور على معرف الطلب');
        }

        const response = await fetch(`/request-license/${requestId}`, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'فشل في إلغاء الطلب');
        }

        // إعادة تعيين النموذج
        document.getElementById('licenseRequestForm').reset();
        document.getElementById('submitRequestBtn').style.display = 'inline-block';
        document.getElementById('downloadLicenseBtn').style.display = 'none';
        document.getElementById('cancelRequestBtn').style.display = 'none';
        
        // مسح معرف الطلب
        localStorage.removeItem('licenseRequestId');

        // عرض رسالة نجاح
        document.getElementById('successMessage').textContent = 'تم إلغاء الطلب بنجاح';
        document.getElementById('successMessage').style.display = 'block';
        document.getElementById('errorMessage').style.display = 'none';

    } catch (error) {
        document.getElementById('errorMessage').textContent = error.message;
        document.getElementById('errorMessage').style.display = 'block';
        document.getElementById('successMessage').style.display = 'none';
    }
}

// إضافة مستمعي الأحداث للأزرار
document.getElementById('downloadLicenseBtn').addEventListener('click', downloadLicense);
document.getElementById('cancelRequestBtn').addEventListener('click', cancelRequest); 