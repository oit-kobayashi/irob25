extends Area2D


# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	var v = Vector2(1, 0).rotated(rotation)
	if Input.is_action_pressed("FORWARD"):
		position += v * 50 * delta
	if Input.is_action_pressed("ROT_CW"):
		rotation += 1 * delta
	if Input.is_action_pressed("ROT_CCW"):
		rotation += -1 * delta
	
	pass
