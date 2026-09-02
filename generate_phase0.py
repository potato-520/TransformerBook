from build_helpers import render_page

# -------------------------------------------------------------
# Chapter 01
# -------------------------------------------------------------
ch01 = """
    <h1>0.1 神经元与感知机 (Neuron & Perceptron)</h1>
    <div class="card">
      <strong>本节目标：</strong>从嵌入式工程师最熟悉的加权电阻分压与比较器（Comparator）电路出发，彻底理解单神经元/感知机的数学本质，掌握 2D 线性可分超平面与决策边界的几何物理意义。
    </div>

    <h2>1. 它解决什么问题？ (工程师视角黑盒引入)</h2>
    <p>
      在嵌入式系统中，我们经常需要处理多路 ADC 采样信号并做出二值判决（例如：根据温度传感器读数 $x_1$ 与电流传感器读数 $x_2$，判定当前系统是否发生过载告警，输出 0 或 1）。
    </p>
    <p>
      <strong>人工神经元（Artificial Neuron）</strong>与<strong>感知机（Perceptron）</strong>正是实现这一目标的最基础单元：
      它接收多个输入信号，分别乘以各自的“权重”（Sensitivity / Gain），进行加权累加后加上一个偏置（Offset / Threshold），最后通过一个阶跃或阈值函数输出 $0$ 或 $1$。
    </p>

    <h2>2. 输入是什么？ (Tensor Shape & 物理含义)</h2>
    <ul>
      <li><strong>输入特征向量 $X$</strong>：<span class="shape-badge">Shape: [B, D_in]</span>。在单个样本场景下，Batch Size $B=1$，$D_{in}=2$（例如 $[x_1, x_2]$ 分别代表特征 1 与特征 2）。数据类型一般为 <code>float32</code>。</li>
      <li><strong>权重向量 $W$</strong>：<span class="shape-badge">Shape: [D_in]</span>。例如 $[w_1, w_2]$，代表每个输入通道对最终决策的贡献权重。</li>
      <li><strong>偏置项 $b$</strong>：<span class="shape-badge">Shape: [1]</span>（标量 float32），决定了决策阈值的高低（基准偏置）。</li>
    </ul>

    <h2>3. 输出是什么？ (Tensor Shape 变换轨迹)</h2>
    <p>
      加权求和（线性部分）：
      $$z = \\sum_{i=1}^{D_{in}} x_i w_i + b = X \\cdot W^T + b \\quad \\text{Shape: } [B, 1]$$
      激活判决（非线性二值化）：
      $$\\hat{y} = \\text{Step}(z) = \\begin{cases} 1, & z \\ge 0 \\\\ 0, & z < 0 \\end{cases} \\quad \\text{Shape: } [B, 1]$$
    </p>

    <h2>4. 黑盒 JS 可交互动画体验</h2>
    <p>
      在下方画布中，红色圆点代表负类样本（$y=0$），绿色方形代表正类样本（$y=1$）。<br/>
      请尝试拖动下方滑块调节权重 $w_1, w_2$ 以及偏置 $b$，观察决策直线 $w_1 x_1 + w_2 x_2 + b = 0$ 的<strong>旋转和平移</strong>，并观察分类准确率的实时变化！
    </p>

    <div class="interactive-card">
      <div class="interactive-title">🎮 2D 线性感知机决策边界动态控制台</div>
      
      <div class="control-panel">
        <div class="control-group">
          <label>权重 $w_1$:</label>
          <input type="range" id="w1_slider" min="-3" max="3" step="0.1" value="1.0">
          <span id="w1_val" style="font-family: Consolas; font-weight: bold; width: 36px;">1.0</span>
        </div>
        <div class="control-group">
          <label>权重 $w_2$:</label>
          <input type="range" id="w2_slider" min="-3" max="3" step="0.1" value="1.0">
          <span id="w2_val" style="font-family: Consolas; font-weight: bold; width: 36px;">1.0</span>
        </div>
        <div class="control-group">
          <label>偏置 $b$:</label>
          <input type="range" id="b_slider" min="-5" max="5" step="0.1" value="-4.2">
          <span id="b_val" style="font-family: Consolas; font-weight: bold; width: 36px;">-4.2</span>
        </div>
        <button class="btn" onclick="resetPerceptron()">重置参数</button>
      </div>

      <div style="display: flex; gap: 20px; align-items: flex-start; flex-wrap: wrap;">
        <canvas id="perceptronCanvas" width="500" height="360" style="border: 1px solid #cbd5e1; border-radius: 8px; background: #fafafa;"></canvas>
        <div style="flex: 1; min-width: 240px;">
          <div class="logic-box" style="margin-top: 0;">
            <strong>实时判定方程：</strong><br/>
            <code id="formula_display">1.00*x1 + 1.00*x2 + (-4.20) = 0</code><br/><br/>
            <strong>当前分类状态：</strong><br/>
            当前准确率：<span id="acc_display" style="font-size: 18px; font-weight: bold; color: #166534;">100%</span><br/>
            法向量方向：<span id="normal_vector" style="font-family: Consolas;">(1.0, 1.0)</span>
          </div>
          <p style="font-size: 13.5px; color: #64748b;">
            💡 <strong>观察技巧</strong>：法向量 $\\vec{w}=(w_1, w_2)$ 垂直于决策直线，指向预测输出为 $1$ 的半平面；偏置 $b$ 控制直线距离原点的垂直距离。
          </p>
        </div>
      </div>
    </div>

    <script>
      const canvas = document.getElementById('perceptronCanvas');
      const ctx = canvas.getContext('2d');

      const samples = [
        {x: 1.0, y: 1.0, label: 0},
        {x: 1.5, y: 0.5, label: 0},
        {x: 0.8, y: 1.8, label: 0},
        {x: 2.0, y: 1.2, label: 0},
        {x: 3.0, y: 2.8, label: 1},
        {x: 2.5, y: 3.5, label: 1},
        {x: 3.8, y: 2.2, label: 1},
        {x: 4.0, y: 3.2, label: 1},
      ];

      function toScreen(x, y) {
        return {
          sx: 40 + x * 80,
          sy: canvas.height - (40 + y * 60)
        };
      }

      function draw() {
        const w1 = parseFloat(document.getElementById('w1_slider').value);
        const w2 = parseFloat(document.getElementById('w2_slider').value);
        const b = parseFloat(document.getElementById('b_slider').value);

        document.getElementById('w1_val').innerText = w1.toFixed(1);
        document.getElementById('w2_val').innerText = w2.toFixed(1);
        document.getElementById('b_val').innerText = b.toFixed(1);
        document.getElementById('formula_display').innerText = `${w1.toFixed(2)}*x1 + ${w2.toFixed(2)}*x2 + (${b.toFixed(2)}) = 0`;
        document.getElementById('normal_vector').innerText = `(${w1.toFixed(2)}, ${w2.toFixed(2)})`;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // 绘制坐标轴
        ctx.strokeStyle = '#e2e8f0';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 5; i++) {
          const pt1 = toScreen(i, 0);
          const pt2 = toScreen(i, 5);
          ctx.beginPath(); ctx.moveTo(pt1.sx, pt1.sy); ctx.lineTo(pt2.sx, pt2.sy); ctx.stroke();
          const pt3 = toScreen(0, i);
          const pt4 = toScreen(5, i);
          ctx.beginPath(); ctx.moveTo(pt3.sx, pt3.sy); ctx.lineTo(pt4.sx, pt4.sy); ctx.stroke();
        }

        // 绘制样本点与准确率评估
        let correct = 0;
        samples.forEach(s => {
          const z = w1 * s.x + w2 * s.y + b;
          const pred = z >= 0 ? 1 : 0;
          if (pred === s.label) correct++;

          const pt = toScreen(s.x, s.y);
          ctx.beginPath();
          if (s.label === 1) {
            ctx.fillStyle = pred === 1 ? '#10b981' : '#f87171';
            ctx.fillRect(pt.sx - 6, pt.sy - 6, 12, 12);
          } else {
            ctx.fillStyle = pred === 0 ? '#ef4444' : '#60a5fa';
            ctx.arc(pt.sx, pt.sy, 6, 0, Math.PI * 2);
            ctx.fill();
          }
          ctx.strokeStyle = '#0f172a';
          ctx.stroke();
        });

        const acc = Math.round((correct / samples.length) * 100);
        document.getElementById('acc_display').innerText = `${acc}% (${correct}/${samples.length})`;
        document.getElementById('acc_display').style.color = acc === 100 ? '#166534' : '#dc2626';

        // 绘制决策直线: w1 * x + w2 * y + b = 0 => y = (-w1 * x - b) / w2
        ctx.strokeStyle = '#2563eb';
        ctx.lineWidth = 3;
        ctx.beginPath();
        if (Math.abs(w2) > 0.001) {
          const y0 = (-w1 * 0 - b) / w2;
          const y5 = (-w1 * 5 - b) / w2;
          const p0 = toScreen(0, y0);
          const p5 = toScreen(5, y5);
          ctx.moveTo(p0.sx, p0.sy);
          ctx.lineTo(p5.sx, p5.sy);
        } else if (Math.abs(w1) > 0.001) {
          const xVal = -b / w1;
          const p0 = toScreen(xVal, 0);
          const p5 = toScreen(xVal, 5);
          ctx.moveTo(p0.sx, p0.sy);
          ctx.lineTo(p5.sx, p5.sy);
        }
        ctx.stroke();
      }

      function resetPerceptron() {
        document.getElementById('w1_slider').value = 1.0;
        document.getElementById('w2_slider').value = 1.0;
        document.getElementById('b_slider').value = -4.2;
        draw();
      }

      document.getElementById('w1_slider').addEventListener('input', draw);
      document.getElementById('w2_slider').addEventListener('input', draw);
      document.getElementById('b_slider').addEventListener('input', draw);
      resetPerceptron();
    </script>

    <h2>5. 极小规模人工计算 (2 维向量手算)</h2>
    <p>
      为了彻底消除纸上手算卡顿，我们使用极小的 2 维向量进行纯数值走查：
    </p>
    <div class="logic-box">
      <strong>已知条件：</strong><br/>
      输入向量：$X = \\begin{bmatrix} 2.0 & -1.0 \\end{bmatrix}$<br/>
      权重参数：$W = \\begin{bmatrix} 0.5 & 1.5 \\end{bmatrix}$，偏置项：$b = -0.2$<br/><br/>
      <strong>计算过程：</strong><br/>
      1. 点积加权和：
      $$z = (x_1 \\cdot w_1) + (x_2 \\cdot w_2) + b = (2.0 \\times 0.5) + (-1.0 \\times 1.5) + (-0.2)$$
      $$z = 1.0 - 1.5 - 0.2 = -0.7$$
      2. 阶跃激活输出：
      $$\\hat{y} = \\text{Step}(-0.7) = 0 \\quad (\\text{因为 } -0.7 < 0)$$
    </div>

    <h2>6. Python / NumPy 极简代码实现</h2>
    <pre><code class="language-python">import numpy as np

def perceptron(x: np.ndarray, w: np.ndarray, b: float):
    # 1. 向量点积加偏置 (MAC 运算)
    z = np.dot(x, w) + b
    
    # 2. 阈值判定
    y_hat = 1 if z >= 0 else 0
    return y_hat, float(z)

# 验证我们刚才的手算例子
x_test = np.array([2.0, -1.0])
w_test = np.array([0.5, 1.5])
b_test = -0.2

y_pred, z_val = perceptron(x_test, w_test, b_test)
print(f"输入点积 z = {z_val:.2f}, 感知机判决输出 = {y_pred}")
# 输出: 输入点积 z = -0.70, 感知机判决输出 = 0
</code></pre>

    <h2>7. 数学原理与物理直觉</h2>
    <p>
      从几何空间来看，方程 $\\vec{w} \\cdot \\vec{x} + b = 0$ 在二维平面上是一条直线，在三维空间中是一个平面，在 $N$ 维空间中则称为<span class="mark-concept">超平面（Hyperplane）</span>：
    </p>
    <ul>
      <li><strong>法向量 $\\vec{w}$</strong>：决定了超平面的“朝向”或旋转角度。超平面垂直于 $\\vec{w}$。</li>
      <li><strong>偏置 $b$</strong>：决定了超平面距离原点的垂直平移距离。原点到超平面的距离为 $d = \\frac{|b|}{\\|\\vec{w}\\|}$。若 $b=0$，超平面必然穿过坐标原点。</li>
    </ul>

    <h2>8. 打开黑盒：真实工程细节 (MCU/CUDA/SIMD)</h2>
    <div class="card">
      <strong>底层硬件对应关系：</strong>
      <ul>
        <li>在 <strong>ARM Cortex-M / MCU</strong> 处理器上，$\\sum x_i w_i + b$ 对应硬件中的 <strong>MAC (Multiply-Accumulate)</strong> 或 <strong>SMLABB</strong> 单周期乘累加汇编指令。</li>
        <li>在 <strong>GPU / CUDA</strong> 架构中，这是最底层的 <strong>FMA (Fused Multiply-Add, 如 <code>__fmaf_rn</code>)</strong> 操作，现代 Tensor Core 能在一个时钟周期内完成数十次此类矩阵点积。</li>
      </ul>
    </div>

    <h2>9. 动手实验与思考题</h2>
    <div class="warning-box">
      <strong>思考题：单感知机的致命短板——异或（XOR）问题</strong><br/>
      若有 4 个样本点：$(0,0)\\to 0, (1,1)\\to 0, (1,0)\\to 1, (0,1)\\to 1$。<br/>
      请问你能否在 2D 平面上画出<strong>一条单一的直线</strong>将 0 和 1 完全分开？<br/>
      <em>提示：这就是 1969 年明斯基指出的感知机局限性，也是促使人们引入多层感知机（MLP）的根本动力！</em>
    </div>
"""

# Render all Phase 0 files
render_page("01_neuron_and_perceptron.html", "0.1 神经元与感知机", "0.1", "00_preface.html", "00 序言", "02_mlp_matrix_representation.html", "0.2 MLP与矩阵表达", ch01)
render_page("02_mlp_matrix_representation.html", "0.2 MLP与矩阵表达", "0.2", "01_neuron_and_perceptron.html", "0.1 神经元与感知机", "03_activation_functions.html", "0.3 激活函数的本质", ch02_body)
render_page("03_activation_functions.html", "0.3 激活函数的本质", "0.3", "02_mlp_matrix_representation.html", "0.2 MLP与矩阵表达", "04_loss_and_cross_entropy.html", "0.4 Loss与交叉熵", ch03_body)
render_page("04_loss_and_cross_entropy.html", "0.4 Loss测量尺与交叉熵", "0.4", "03_activation_functions.html", "0.3 激活函数的本质", "05_gradients_and_backpropagation.html", "0.5 梯度下降与反传", ch04_body)
render_page("05_gradients_and_backpropagation.html", "0.5 梯度下降与反向传播", "0.5", "04_loss_and_cross_entropy.html", "0.4 Loss与交叉熵", "06_tokenization_and_bpe.html", "1.1 Tokenization与BPE", ch05_body)
print("Phase 0 generated successfully!")
