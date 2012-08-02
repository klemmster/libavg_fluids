#ifndef PROCESSTHREAD_H_NJD6PWKX
#define PROCESSTHREAD_H_NJD6PWKX

#include <base/WorkerThread.h>
#include <base/Queue.h>

typedef boost::shared_ptr<boost::mutex> MutexPtr;
namespace avg
{

    class ProcessThread: public WorkerThread<ProcessThread>
    {
    public:
        ProcessThread (CQueue& CmdQ, const std::string& threadName, MutexPtr pMutex);
        virtual ~ProcessThread ();

        bool init();
        void deinit();
        bool work();

    protected:

    private:
        MutexPtr m_pMutex;
        bool m_doWork;
    };

} /* avg */

#endif /* end of include guard: PROCESSTHREAD_H_NJD6PWKX */

