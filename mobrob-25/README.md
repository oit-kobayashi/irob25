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
- GDScript にキーワード引数はないらしい

## 12/23 to-do memo
- Timer で予測更新コールバックを作る
- _process 内の dsr, dsl を別の名前にリファクタリングして dsr, dsl はその積算ようとして(グローバルスコープで)使用。
- 前項の dsr, dsl を Timer callback で使用。Σもグローバルスコープに作る。
- _draw() 関数を追加して楕円を描く

## 1/6 to-do memo
- _draw() 関数を追加して楕円を描く (12/23 のやり残し)
- draw_set_transform_matrix() と draw_circle() で楕円が描ける
- 楕円の傾きは固有ベクトル (lambda, P = np.linalg.eig()) から計算
- robot = {
      'id': id,
      'pose': init_val.pose,
      'sigma': sigma,
      'eigenvalues': eigenvalues,    # 追加
      'eigenvectors': eigenvectors   # 追加
  }
- 注意: eig() で求めた P において固有ベクトルは縦ベクトルなので robot に格納する際は適切に取り出す必要あり

## 1/13 memo
- root ノードは Node や Node2D。
- 壁は StaticBody2D が良いらしい。
- robot が body_entered を発火。
- [issue] 衝突対策は move_and_slide() を使うべきらしい。その場合 _process ではなく _physics_process にする必要がある。
