async function sendRequest(endpoint, data) {
    try {
        const response = await fetch('http://localhost:3000/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Error:', error);
        alert('发生错误！请检查控制台获取更多信息。');
    }
}



function getFormData() {
    const form = document.getElementById('parametersForm');
    const data = {};
    for (let element of form.elements) {
        if (element.type !== 'button' && element.value) {
            data[element.id] = parseFloat(element.value);
        }
    }
    return Object.keys(data).length ? data : null;
}

function updateProgress(percent) {
    const progressBarFill = document.getElementById('progressBarFill');
    progressBarFill.style.width = `${percent}%`;
    progressBarFill.textContent = `${percent}%`;
}


function saveParams() {
    var form = document.getElementById('parametersForm');
    var data = {};

    
    // 遍历表单中的所有输入项
    for (var i = 0; i < form.elements.length; i++) {
      var element = form.elements[i];
      // 检查元素是否为输入项并且有值
      if (element.type !== 'button' && element.value) {
        data[element.name] = element.value;

      }
    }
    
    // 将参数对象转换为字符串并存储在localStorage中
    localStorage.setItem('laserRemoteSensingParams', JSON.stringify(data));
    alert('参数已保存！');
    document.getElementById('startSimulation').addEventListener('click', function() {
        const fileInput = document.getElementById('fileInput');
        fileInput.click(); // 模拟点击文件选择器
    
        fileInput.addEventListener('change', function() {
            const files = fileInput.files;
            if (files.length === 0) {
                alert('请选择文件后再点击上传！');
                return;
            }
    
            const formData = new FormData();
            for (const file of files) {
                formData.append('files[]', file);
            }
    
            const xhr = new XMLHttpRequest();
            xhr.open('POST', '/upload', true);
    
            xhr.upload.onprogress = function(event) {
                if (event.lengthComputable) {
                    const percentComplete = Math.round((event.loaded / event.total) * 100);
                    const progressBar = document.getElementById('progressBar');
                    progressBar.style.width = percentComplete + '%';
                    progressBar.textContent = percentComplete + '%';
                    document.getElementById('progressContainer').style.display = 'block';
                }
            };
    
            xhr.onload = function() {
                if (xhr.status === 200) {
                    alert('文件上传成功！');
                } else {
                    alert('文件上传失败！');
                }
                document.getElementById('progressContainer').style.display = 'none';
            };
    
            xhr.send(formData);
        });
    });
        
    var fileInput = document.getElementById('fileInput');
    var fileNames = [];

    // 检查是否有文件被选中
    if (fileInput.files.length > 0) {
        // 遍历所有选择的文件，获取文件名
        for (var i = 0; i < fileInput.files.length; i++) {
            fileNames.push(fileInput.files[i].name); // 获取每个文件的文件名
        }

        // 将文件名数组保存到 localStorage
        localStorage.setItem('CAL_fileName', JSON.stringify(fileNames));
    
    alert('文件名已记录: ' + fileNames.join(', '));
} else {
    alert('请选择文件');
}
const requestData = {
    data: data,
    fileNames: fileNames
};
    fetch('/saveConfig', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
        
    })
    .then(response => response.json())
    .then(result => {
        alert('参数已保存！');
    })
    .catch(error => {
        console.error('保存参数时发生错误:', error);
    });
  }
  function triggerFileInput() {
    document.getElementById('fileInput').click(); // 点击按钮时触发文件选择框
}

// 记录文件名并保存到 localStorage


function startSimulation() {
    
    fetch('/startSimulation', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        console.log('Simulation started:', data);
        
        document.getElementById('plot_image1').src = '/plot_P.png'; 
    })
    .catch(error => {
        console.error('Error starting simulation:', error);
    });
}

function startRetrieval() {

    fetch('/startRetrieval', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        console.log('Retrieval started:', data);
        
        document.getElementById('plot_image2').src = '/plot_r.png'; 
    })
    .catch(error => {
        console.error('Error starting retrieval:', error);
    });
}

function startFusion() {
    
    fetch('/startFusion', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        console.log('Fusion started:', data);
        
        document.getElementById('plot_image3').src = '/plot_DAOD.png'; 
    })
    .catch(error => {
        console.error('Error starting fusion:', error);
    });
}
window.onload = function() {
    var preElement = document.getElementById("content_r");
    var newContent = ""; // 新的内容
    preElement.textContent = newContent;
    // 保存修改后的内容到 localStorage
    localStorage.setItem("modifiedContent", newContent);
    var savedContent = localStorage.getItem("modifiedContent");
    if (savedContent) {
        preElement.innerHTML = savedContent; // 如果有保存的内容，则显示
    }
};
