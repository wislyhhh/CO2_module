const express = require('express');
const bodyParser = require('body-parser');
const { exec } = require('child_process');
const path = require('path');
const fs = require('fs'); // 导入fs模块
const app = express();
const port = 3000; // 初始端口号

// 中间件 - 静态文件
app.use(express.json()); // 确保 express 能解析 JSON 请求体

app.use(express.static(path.join(__dirname, 'public')));

// 解析 JSON 请求体
app.use(bodyParser.json());

function startServer(port) {
    const server = app.listen(port, () => {
        console.log(`Server is running at http://localhost:${port}`);
    });

    server.on('error', (err) => {
        if (err.code === 'EADDRINUSE') {
            console.error(`Port ${port} is already in use. Trying another port...`);
            startServer(port + 1); // 尝试下一个端口
        } else {
            console.error(err);
        }
    });
}

startServer(port);

// 启动仿真端点
app.post('/startSimulation', (req, res) => {
    exec('python ./runSimulation.py', (error, stdout, stderr) => {
        if (error) {
            console.error(`exec error: ${error}`);
            return res.status(500).json({ message: '程序出错！' });
        }
        console.log(`stdout: ${stdout}`);
        res.json({ progress: 100, output: stdout });
    });
});

// 启动检索端点
app.post('/startRetrieval', (req, res) => {
    exec('python ./runRetrieval.py', (error, stdout, stderr) => {
        if (error) {
            console.error(`exec error: ${error}`);
            return res.status(500).json({ message: '程序出错！' });
        }
        console.log(`stdout: ${stdout}`);
        res.json({ progress: 100, output: stdout });
    });
});

// 启动融合端点
app.post('/startFusion', (req, res) => {
    exec('python ./runFusion.py', (error, stdout, stderr) => {
        if (error) {
            console.error(`exec error: ${error}`);
            return res.status(500).json({ message: '程序出错！' });
        }
        console.log(`stdout: ${stdout}`);
        res.json({ progress: 100 , output: stdout });
    });
});

// 保存配置端点
app.post('/saveConfig', (req, res) => {
    const { data, fileNames } = req.body;
    if (!data || !fileNames) {
        return res.status(400).json({ message: 'Invalid data or fileNames' });
    }console.log('Received data:', data);
    console.log('Received fileNames:', fileNames);
    

    const configContent = `[NEWSETTINGS]
energy_off=${data.energy_off}
energy_on=${data.energy_on}
optical_efficient=${data.optical_efficient}
pulse_width=${data.pulse_width}
reference_wavelength=${data.reference_wavelength}
satellite_altitude=${data.satellite_altitude}
working_wavelength=${data.working_wavelength}
FOV=${data.FOV}
excess_noise_factor=${data.excess_noise_factor}
filter_bandwidth=${data.filter_bandwidth}
intergal_G_factor=${data.intergal_G_factor}
responsivity=${data.responsivity}
telescope_diam=${data.telescope_diam}
day_night_flag=${data.day_night_flag || 0}
CAL_fileName=${fileNames.join(', ')}
`;

    fs.writeFile(path.join(__dirname, 'input', 'Config_Parameters.ini'), configContent, (err) => {
        if (err) {
            console.error('Failed to write config file:', err);
            return res.status(500).json({ message: 'Failed to save configuration' });
        }
        res.json({ message: 'Configuration saved successfully' });
    });
});
