# self localization simulation of mobile robots

## フロント(godot)とバック(サーバ)の役割

### 案1 バックはその場でのstateless計算に徹する
- フロント側がロボットのすべての情報を持つ
- バックはstatelessに計算リクエストをこなすだけ
- シミュレーションの時間ステップを進めるのはフロント

### 案2 フロントはUIに徹する
- フロントはロボットの状態を一切持たず、statelessに描画するだけ
- バックが全ての情報を持ち、計算も行う
- シミュレーションの時間ステップを進めるのはフロント
- バックエンドに DB が必要

### 案3 真値はフロント、推定値はバック
- フロントは実際のロボットの状態を管理
- バックは自己位置 *推定* に関する一切を管理
- シミュレーションの時間ステップを進めるのはフロント
- バックエンドに DB が必要


## バック仕様
- REST + RPC ぽい感じ
- 案1の例: 
  + endpoint: /move_robot
  + method: POST
  + params:{"pose": [[1],[2],[3]], "dsl": 4, "dsr": 5}
  + return: {"pose": [[7],[8],[9]]}
- 案2, 3の例: 
  + endpoint: /robot/{id}
  + method: POST
  + params: {method: "move", "dsl": 4, "dsr": 5}
  + return: 200 OK のみ (位置は GET /robot/{id} で)

## memo
- HTTPie というモジュールをインストールすると http というコマンドでJSON postが楽
