import json
import shutil
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

class SessionViewer:
    def __init__(self, root_dir, fps=15.0):
        self.root_dir = Path(root_dir)
        self.sessions = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        self.current_idx = 0
        self.fps = fps
        self.running = True
        self.skip_session = False
        self.paused = False
        
        if not self.sessions:
            print("未找到任何轨迹目录")
            return

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        self.ax.axis("off")
        self.im_artist = None
        
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self._update_title()

    def _update_title(self):
        if not self.sessions:
            self.fig.canvas.manager.set_window_title("No Sessions Left")
            return
        state = "PAUSED" if self.paused else "PLAYING"
        name = self.sessions[self.current_idx].name
        title = f"[{state}] J:Prev | K:Next | P:Pause | D:DELETE(Direct) | {self.current_idx + 1}/{len(self.sessions)} - {name}"
        self.fig.canvas.manager.set_window_title(title)

    def _load_steps(self, session_path):
        p = session_path / "steps.jsonl"
        return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()] if p.exists() else []

    def _load_frame(self, step, session_dir):
        meta = step.get("rgb", {})
        path = session_dir / meta.get("path", "")
        if not path.exists(): 
            raise FileNotFoundError(f"Missing file: {path}")
        
        b = path.read_bytes()
        w, h = int(meta["width"]), int(meta["height"])
        expected_size = w * h * 4
        if len(b) < expected_size:
            raise ValueError(f"Buffer size mismatch: {len(b)} < {expected_size}")
            
        arr = np.frombuffer(b, dtype=np.uint8)[:expected_size].reshape((h, w, 4))
        return Image.fromarray(arr[:, :, :3])

    def _on_key(self, event):
        if event.key == 'k':
            self.current_idx = (self.current_idx + 1) % len(self.sessions)
            self.skip_session = True
            self.paused = False
        elif event.key == 'j':
            self.current_idx = (self.current_idx - 1) % len(self.sessions)
            self.skip_session = True
            self.paused = False
        elif event.key == 'p':
            self.paused = not self.paused
            self._update_title()
        elif event.key == 'd':
            if self.sessions:
                target = self.sessions.pop(self.current_idx)
                shutil.rmtree(target) # 直接删除，不再确认
                print(f"已删除: {target.name}")
                if not self.sessions:
                    self.running = False
                else:
                    self.current_idx %= len(self.sessions)
                self.skip_session = True
                self.paused = False

    def _update_terminal_info(self):
        """在终端输出当前播放状态"""
        if not self.sessions:
            return
        curr = self.sessions[self.current_idx]
        print("\n" + "="*50)
        print(f"正在播放 ({self.current_idx + 1}/{len(self.sessions)}):")
        print(f"目录名: {curr.name}")
        print(f"完整路径: {curr}")
        print("="*50)

    def run(self):
        while self.running and self.sessions:
            self.skip_session = False
            curr_session = self.sessions[self.current_idx]
            try:
                steps = self._load_steps(curr_session)
                self._update_title()
                self._update_terminal_info()
                
                self.ax.cla()
                self.ax.axis("off")
                self.im_artist = None

                for i, step in enumerate(steps):
                    while self.paused and not self.skip_session:
                        plt.pause(0.1)
                    
                    if self.skip_session: break
                    
                    img = self._load_frame(step, curr_session)
                    if img is None: continue
                    
                    # 简单信息标注
                    draw = ImageDraw.Draw(img)
                    draw.text((10, 10), f"STEP: {i}/{len(steps)}", fill=(255, 255, 0))
                    
                    frame = np.asarray(img)
                    if self.im_artist is None:
                        self.im_artist = self.ax.imshow(frame)
                    else:
                        self.im_artist.set_data(frame)
                    
                    self.fig.canvas.draw_idle()
                    plt.pause(1.0 / self.fps)
            except (FileNotFoundError, ValueError, RuntimeError) as e:
                print(f"\n[错误] 轨迹数据损坏: {curr_session.name}")
                print(f"原因: {e}")
                
                # 执行自动删除逻辑
                if curr_session.exists():
                    shutil.rmtree(curr_session)
                    print(f"已自动清除损坏的本地文件。")
                
                # 从列表中移除并跳过
                self.sessions.pop(self.current_idx)
                if self.sessions:
                    self.current_idx %= len(self.sessions)
                self.skip_session = True
            if not self.skip_session and self.sessions:
                self.current_idx = (self.current_idx + 1) % len(self.sessions)

        plt.ioff()
        plt.close()
        print("所有轨迹已播完或已删除")

if __name__ == "__main__":
    path = r"E:\ToolApps\Steam\steamapps\common\Grand Theft Auto V\data\manual"
    viewer = SessionViewer(path)
    viewer.run()