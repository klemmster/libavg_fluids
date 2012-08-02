#include "ProcessThread.h"

namespace avg
{

    ProcessThread::ProcessThread(CQueue& CmdQ, const std::string& threadName,
            MutexPtr pMutex):
        WorkerThread<ProcessThread>(threadName, CmdQ),
        m_pMutex(pMutex),
        m_doWork(false)
    {}

    ProcessThread::~ProcessThread(){
    }

    bool ProcessThread::init(){
        m_doWork = true;
        return true;
    }

    void ProcessThread::deinit(){
    }

    bool ProcessThread::work(){
        if(m_doWork){
            m_doWork = false;
        }
        return true;
    }

} /* avg */
