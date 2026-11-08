from flask import Flask, render_template, jsonify, send_from_directory, request, redirect, url_for
import os
import sys
import json
import markdown
import shutil

# 将当前目录添加到系统路径，修复导入问题
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR, OUTPUT_DIR, DB_CONFIG
from run_pipeline import run_full_pipeline
from utils.db_helper import get_engine
import pandas as pd
import zipfile
import io

app = Flask(__name__)

# 配置上传目录
app.config['UPLOAD_FOLDER'] = DATA_DIR

@app.route('/')
def index():
    # 检查当前数据文件
    files = []
    if os.path.exists(DATA_DIR):
        files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    
    return render_template('index.html', files=files)

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return "没有文件上传", 400
    
    uploaded_files = request.files.getlist('files')
    if not uploaded_files or uploaded_files[0].filename == '':
        return "没有选择文件", 400

    # 清空旧数据（可选，根据用户需求）
    # for f in os.listdir(DATA_DIR):
    #     os.remove(os.path.join(DATA_DIR, f))

    for file in uploaded_files:
        if file and file.filename.endswith('.csv'):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            
    return redirect(url_for('index'))

@app.route('/run_pipeline', methods=['POST'])
def trigger_pipeline():
    try:
        threshold = request.form.get('threshold')
        if threshold:
            threshold = float(threshold)
        run_full_pipeline(custom_threshold=threshold)
        return jsonify({"status": "success", "message": "流水线运行成功！"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/output/<path:filename>')
def output_file(filename):
    """提供 output 目录下的图片等静态资源，供报告页面引用"""
    return send_from_directory(OUTPUT_DIR, filename)

@app.route('/show_report')
def show_report():
    report_path = os.path.join(OUTPUT_DIR, 'clean_report.md')
    if not os.path.exists(report_path):
        return "报告尚未生成，请先运行流水线。"
    
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
        html = markdown.markdown(content, extensions=['tables'])

    # 将报告中的相对图片路径改写为 /output/ 路由，保证网页中正常显示
    html = html.replace('src="', 'src="/output/')
    
    return f"""
    <html>
    <head>
        <style>
            body {{ font-family: sans-serif; padding: 20px; line-height: 1.6; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <a href="/">返回首页</a>
        {html}
    </body>
    </html>
    """

@app.route('/api/get_metrics')
def get_metrics():
    metrics_path = os.path.join(OUTPUT_DIR, 'test_true_label_metrics.json')
    if not os.path.exists(metrics_path):
        return jsonify({"error": "Metrics not found"}), 404
    
    with open(metrics_path, 'r') as f:
        data = json.load(f)
    return jsonify(data)

@app.route('/db/show_tables')
def show_tables():
    engine = get_engine()
    tables = [
        'raw_data', 'dataset_check_table', 'label_stat_table',
        'dataset_split_stat', 'split_market_stat',
        'method_compare_result', 'market_f1_result', 'leave_one_out_result',
        'ablation_result', 'clean_quality_compare'
    ]
    
    html = "<h1>数据库表预览</h1><a href='/'>返回首页</a>"
    for table in tables:
        try:
            df = pd.read_sql(f"SELECT * FROM {table} LIMIT 5", engine)
            html += f"<h3>{table} (Top 5)</h3>"
            html += df.to_html(classes='table', index=False)
        except Exception as e:
            html += f"<h3>{table}</h3><p>无法读取表或表不存在: {str(e)}</p>"
            
    return f"""
    <html>
    <head>
        <style>
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>{html}</body>
    </html>
    """

@app.route('/download/output')
def download_output():
    # 将 output 目录打包
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for file in files:
                zf.write(os.path.join(root, file), file)
    memory_file.seek(0)
    
    from flask import send_file
    return send_file(memory_file, download_name='output.zip', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
