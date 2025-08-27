from Module import stacking,draw_skidder,process

token = process(r"Data/output.json", "privilege")
load = stacking(token,"privilege",20,0.8)
draw_skidder(load)