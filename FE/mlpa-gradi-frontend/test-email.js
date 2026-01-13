const waitOn = require('wait-on');
const { exec } = require('child_process');

const opts = {
    resources: [
        'http://localhost:8080/api/email/verification-code',
    ],
    delay: 1000,
    interval: 500,
    timeout: 30000,
    validateStatus: function (status) {
        return status === 405; // GET is usually 405 for POST-only endpoints
    },
};

console.log('Waiting for Spring Boot server to be ready...');

waitOn(opts)
    .then(() => {
        console.log('Server is ready! Running tests...');

        // 1. Send verification code
        const sendCmd = `powershell -Command "Invoke-RestMethod -Uri 'http://localhost:8080/api/email/verification-code' -Method Post -ContentType 'application/json' -Body (Get-Content '../req.json' -Raw)"`;

        exec(sendCmd, (err, stdout, stderr) => {
            if (err) {
                console.error('Error sending code:', stderr);
                process.exit(1);
            }
            console.log('Send Code Response:', stdout);

            console.log('Please check the server console for the generated 6-digit code.');
            console.log('Format: Generated code for student 32204077: XXXXXX');
        });
    })
    .catch((err) => {
        console.error('Wait-on error:', err);
        process.exit(1);
    });
