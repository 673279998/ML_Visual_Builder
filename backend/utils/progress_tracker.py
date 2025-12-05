"""
训练进度追踪器
用于追踪长时间运行的训练任务的进度
"""
import threading
import time
from typing import Dict, Any, Optional
from datetime import datetime


class ProgressTracker:
    """进度追踪器 - 单例模式"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.tasks = {}  # 存储所有任务的进度信息
        self._initialized = True
    
    def create_task(self, task_id: str, task_type: str, total_steps: int = 100) -> None:
        """
        创建新任务
        
        Args:
            task_id: 任务ID
            task_type: 任务类型 (training, tuning, prediction等)
            total_steps: 总步骤数
        """
        self.tasks[task_id] = {
            'task_id': task_id,
            'task_type': task_type,
            'status': 'running',  # running, completed, failed, cancelled
            'progress': 0,  # 进度百分比 0-100
            'current_step': 0,
            'total_steps': total_steps,
            'message': '任务开始...',
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'error': None,
            'metadata': {}  # 额外的元数据
        }
    
    def update_progress(self, task_id: str, progress: int = None, 
                       current_step: int = None, message: str = None,
                       metadata: Dict[str, Any] = None) -> None:
        """
        更新任务进度
        
        Args:
            task_id: 任务ID
            progress: 进度百分比 (0-100)
            current_step: 当前步骤
            message: 状态消息
            metadata: 额外元数据
        """
        if task_id not in self.tasks:
            return
        
        task = self.tasks[task_id]
        
        if progress is not None:
            task['progress'] = min(100, max(0, progress))
        
        if current_step is not None:
            task['current_step'] = current_step
            # 根据当前步骤自动计算进度
            if task['total_steps'] > 0:
                task['progress'] = int((current_step / task['total_steps']) * 100)
        
        if message is not None:
            task['message'] = message
        
        if metadata is not None:
            task['metadata'].update(metadata)
    
    def complete_task(self, task_id: str, message: str = '任务完成', 
                     result: Any = None) -> None:
        """
        标记任务完成
        
        Args:
            task_id: 任务ID
            message: 完成消息
            result: 任务结果
        """
        if task_id not in self.tasks:
            return
        
        task = self.tasks[task_id]
        task['status'] = 'completed'
        task['progress'] = 100
        task['message'] = message
        task['end_time'] = datetime.now().isoformat()
        
        if result is not None:
            task['metadata']['result'] = result
    
    def fail_task(self, task_id: str, error: str) -> None:
        """
        标记任务失败
        
        Args:
            task_id: 任务ID
            error: 错误信息
        """
        if task_id not in self.tasks:
            return
        
        task = self.tasks[task_id]
        task['status'] = 'failed'
        task['message'] = '任务失败'
        task['error'] = error
        task['end_time'] = datetime.now().isoformat()
    
    def cancel_task(self, task_id: str) -> None:
        """
        取消任务
        
        Args:
            task_id: 任务ID
        """
        if task_id not in self.tasks:
            return
        
        task = self.tasks[task_id]
        task['status'] = 'cancelled'
        task['message'] = '任务已取消'
        task['end_time'] = datetime.now().isoformat()
    
    def get_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务进度
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务进度信息，如果任务不存在则返回None
        """
        return self.tasks.get(task_id)
    
    def get_all_tasks(self, task_type: str = None) -> list:
        """
        获取所有任务
        
        Args:
            task_type: 任务类型过滤 (可选)
            
        Returns:
            任务列表
        """
        if task_type:
            return [task for task in self.tasks.values() if task['task_type'] == task_type]
        return list(self.tasks.values())
    
    def cleanup_completed_tasks(self, max_age_seconds: int = 3600) -> int:
        """
        清理已完成的旧任务
        
        Args:
            max_age_seconds: 最大保留时间（秒）
            
        Returns:
            清理的任务数量
        """
        now = datetime.now()
        removed_count = 0
        task_ids_to_remove = []
        
        for task_id, task in self.tasks.items():
            if task['status'] in ['completed', 'failed', 'cancelled']:
                if task['end_time']:
                    end_time = datetime.fromisoformat(task['end_time'])
                    age = (now - end_time).total_seconds()
                    if age > max_age_seconds:
                        task_ids_to_remove.append(task_id)
        
        for task_id in task_ids_to_remove:
            del self.tasks[task_id]
            removed_count += 1
        
        return removed_count
    
    def delete_task(self, task_id: str) -> bool:
        """
        删除指定任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否成功删除
        """
        if task_id in self.tasks:
            del self.tasks[task_id]
            return True
        return False


# 全局进度追踪器实例
progress_tracker = ProgressTracker()
