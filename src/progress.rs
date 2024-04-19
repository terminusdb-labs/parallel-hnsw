use std::any::Any;

pub struct ProgressUpdate {}
pub struct Interrupt;

pub trait ProgressMonitor: Send {
    fn update(&mut self, update: ProgressUpdate) -> Result<(), Interrupt>;
    fn keep_alive(&mut self) -> Box<dyn Any>;
}

impl ProgressMonitor for () {
    fn update(&mut self, _update: ProgressUpdate) -> Result<(), Interrupt> {
        Ok(())
    }

    fn keep_alive(&mut self) -> Box<dyn Any> {
        Box::new(())
    }
}

impl ProgressMonitor for Box<dyn ProgressMonitor> {
    fn update(&mut self, update: ProgressUpdate) -> Result<(), Interrupt> {
        (**self).update(update)
    }

    fn keep_alive(&mut self) -> Box<dyn Any> {
        (**self).keep_alive()
    }
}

pub fn keep_alive_while<T>(monitor: &mut impl ProgressMonitor, mut f: impl FnMut() -> T) -> T {
    let _guard = monitor.keep_alive();
    f()
}
