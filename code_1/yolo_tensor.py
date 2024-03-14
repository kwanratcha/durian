from flask import Flask, request, jsonify

app = Flask(__name__)

# Define endpoint for image prediction
@app.route('/predict', methods=['POST'])
def predict():
    # โค้ดการรับ request และการประมวลผล
    return jsonify({'predicted_class': 'thongY', 'probability': 0.85})

if __name__ == '__main__':
    app.run(debug=True)
