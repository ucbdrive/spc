#include <Python.h>
#include <arrayobject.h>
#include <sys/shm.h>
#include <spawn.h>
#include <sys/types.h>
#include <sys/wait.h>

#define image_width 640
#define image_height 480

struct shared_use_st
{
    int written;  // a label, if 1: available to read, if 0: available to write
    uint8_t data[image_width * image_height * 3];  // image data field
    uint8_t data_remove_side[image_width * image_height * 3];
    uint8_t data_remove_middle[image_width * image_height * 3];
    uint8_t data_remove_car[image_width * image_height * 3];
    int pid;
    int isEnd;
    double dist; // distance covered

    double steerCmd;
    double accelCmd;
    double brakeCmd;
    // for reward building
    double speed;
    double angle_in_rad;
    int damage;
    double pos;
    int segtype;
    double radius;
    int frontCarNum;
    double frontDist;

    double posX;
    double posY;
    double posZ;

    double width;
};

static PyObject * mem_init(PyObject *self, PyObject *args)
{
    int mkey;
    if (!PyArg_ParseTuple(args, "i", &mkey))
        return NULL;
    // set up memory sharing
    int shmid = shmget((key_t)mkey, sizeof(struct shared_use_st), 0666 | IPC_CREAT);
    if(shmid == -1)
    {
        PyErr_SetString(PyExc_MemoryError, "shmget failed");
        return NULL;
    }
    // printf("********** Controler: Set memory sharing key to %d successfully **********\n", mkey);

    void *shm = shmat(shmid, 0, 0);
    if(shm == (void*)-1)
    {
        PyErr_SetString(PyExc_MemoryError, "shmat failed");
        return NULL;
    }
    // printf("********** Controler: Memory sharing started, attached at 0x%08X **********\n", *((int*)(shm)));

    struct shared_use_st* shared = (struct shared_use_st*)shm;
    memset(shared, 0, sizeof(struct shared_use_st));
    shared->written = 0;
    shared->pid = 0;
    shared->isEnd = 0;

    shared->steerCmd = 0.0;
    shared->accelCmd = 0.0;
    shared->brakeCmd = 0.0;

    return Py_BuildValue("il", shmid, (long)shm);
}

static PyObject * mem_clear(PyObject *self, PyObject *args)
{
    void *shm;
    if (!PyArg_ParseTuple(args, "l", &shm))
        return NULL;

    struct shared_use_st* shared = (struct shared_use_st*)shm;
    memset(shared, 0, sizeof(struct shared_use_st));
    shared->written = 0;
    shared->pid = 0;
    shared->isEnd = 0;

    shared->steerCmd = 0.0;
    shared->accelCmd = 0.0;
    shared->brakeCmd = 0.0;

    Py_RETURN_NONE;
}

static PyObject * mem_cleanup(PyObject *self, PyObject *args)
{
    int shmid;
    void *shm;
    if (!PyArg_ParseTuple(args, "il", &shmid, &shm))
        return NULL;

    if (shmdt(shm) == -1)
    {
        PyErr_SetString(PyExc_MemoryError, "shmdt failed");
        return NULL;
    } 
    else if (shmctl(shmid, IPC_RMID, 0) == -1)
    {
        PyErr_SetString(PyExc_MemoryError, "shmctl(IPC_RMID) failed");
        return NULL;
    }
    
    Py_RETURN_NONE;
}

extern char **environ;
static PyObject * env_start(PyObject *self, PyObject *args)
{
    int auto_back, mkey, isServer, display_num;
    char *config_path;
    if (!PyArg_ParseTuple(args, "iiiis", &auto_back, &mkey, &isServer, &display_num, &config_path))
        return NULL;
    int arg_count = 5;
    char *torcs_args[9];
    torcs_args[0] = (char*)"torcs";
    torcs_args[1] = (char*)"_rgs";
    torcs_args[2] = config_path;
    torcs_args[3] = (char*)"_mkey";
    char mkey_str[10];
    sprintf(mkey_str, "%d", mkey);
    torcs_args[4] = mkey_str;
    if (isServer)
    {
        torcs_args[arg_count++] = (char*)"_screen";
        char display_str[10];
        sprintf(display_str, "%d", display_num);
        torcs_args[arg_count++] = display_str;
    }
    if (auto_back)
    {
        torcs_args[arg_count++] = (char*)"_back";
    }
    torcs_args[arg_count++] = NULL;

    pid_t pid;
    posix_spawn(&pid, "/usr/local/bin/torcs", NULL, NULL, torcs_args, environ);
    
    return Py_BuildValue("i", (int)pid);
}

static PyObject * env_action_continuous(PyObject *self, PyObject *args)
{
    void *shm;
    double accel, brake, steer;
    if (!PyArg_ParseTuple(args, "lddd", &shm, &accel, &brake, &steer))
        return NULL;
    
    struct shared_use_st* shared = (struct shared_use_st*)shm;
    shared->accelCmd = accel;
    shared->brakeCmd = brake;
    shared->steerCmd = steer;
    shared->written = 0;
    
    Py_RETURN_NONE;
}

static PyObject * env_terminate(PyObject *self, PyObject *args)
{
    pid_t pid;
    void *shm;
    if (!PyArg_ParseTuple(args, "il", &pid, &shm))
        return NULL;

    struct shared_use_st* shared = (struct shared_use_st*)shm;

    kill(pid, SIGKILL);
    waitpid(pid, NULL, 0);

    kill(shared->pid, SIGKILL);
    waitpid(shared->pid, NULL, 0);
    
    Py_RETURN_NONE;
}

static PyObject * env_get_state(PyObject *self, PyObject *args)
{
    void *shm;
    if (!PyArg_ParseTuple(args, "l", &shm))
        return NULL;
    struct shared_use_st* shared = (struct shared_use_st*)shm;
    int end = shared->isEnd;
    double dist = shared->dist;
    double speed = shared->speed;
    double angle = shared->angle_in_rad;
    int damage = shared->damage;
    double pos = shared->pos;
    int segtype = shared->segtype;
    double radius = shared->radius;
    int frontCarNum = shared->frontCarNum;
    double frontDist = shared->frontDist;
    double posX = shared->posX;
    double posY = shared->posY;
    double posZ = shared->posZ;
    double width = shared->width;

    return Py_BuildValue("idddidididdddd", end, dist, speed, angle, damage, pos, segtype, radius, frontCarNum, frontDist, posX, posY, posZ, width);
}

static PyObject * getRGBImage(PyObject *self, PyObject *args)
{
    void *shm;
    if (!PyArg_ParseTuple(args, "l", &shm))
        return NULL;
    struct shared_use_st* shared = (struct shared_use_st*)shm;

    uint8_t* image = shared->data;
    npy_intp dims[3] = {image_height, image_width, 3};
    PyArrayObject* img = (PyArrayObject *) PyArray_SimpleNewFromData(3, dims, NPY_UINT8, image);

    return Py_BuildValue("O", img);
}

static PyObject * get_segmentation(PyObject *self, PyObject *args)
{
    void *shm;
    if (!PyArg_ParseTuple(args, "l", &shm))
        return NULL;
    struct shared_use_st* shared = (struct shared_use_st*)shm;

    uint8_t* image_remove_side = shared->data_remove_side;
    uint8_t* image_remove_middle = shared->data_remove_middle;
    uint8_t* image_remove_car = shared->data_remove_car;
    npy_intp dims[3] = {image_height, image_width, 3};
    PyArrayObject* img_remove_side = (PyArrayObject *) PyArray_SimpleNewFromData(3, dims, NPY_UINT8, image_remove_side);
    PyArrayObject* img_remove_middle = (PyArrayObject *) PyArray_SimpleNewFromData(3, dims, NPY_UINT8, image_remove_middle);
    PyArrayObject* img_remove_car = (PyArrayObject *) PyArray_SimpleNewFromData(3, dims, NPY_UINT8, image_remove_car);

    return Py_BuildValue("OOO", img_remove_side, img_remove_middle, img_remove_car);
}

static PyObject * get_written(PyObject *self, PyObject *args)
{
    void *shm;
    if (!PyArg_ParseTuple(args, "l", &shm))
        return NULL;
    struct shared_use_st* shared = (struct shared_use_st*)shm;
    return Py_BuildValue("i", shared->written);
}

static PyMethodDef TORCS_ctrl_Methods[] = {
    {"mem_init", mem_init, METH_VARARGS, ""},
    {"mem_cleanup", mem_cleanup, METH_VARARGS, ""},
    {"mem_clear", mem_clear, METH_VARARGS, ""},
    {"env_start", env_start, METH_VARARGS, ""},
    {"env_action_continuous", env_action_continuous, METH_VARARGS, ""},
    {"env_terminate", env_terminate, METH_VARARGS, ""},
    {"env_get_state", env_get_state, METH_VARARGS, ""},
    {"getRGBImage", getRGBImage, METH_VARARGS, ""},
    {"get_segmentation", get_segmentation, METH_VARARGS, ""},
    {"get_written", get_written, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "TORCS_ctrl",
        NULL,
        0,
        TORCS_ctrl_Methods,
        NULL,
        NULL,
        NULL,
        NULL
};

PyMODINIT_FUNC PyInit_TORCS_ctrl(void)
#else
PyMODINIT_FUNC initTORCS_ctrl(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    (void) Py_InitModule("TORCS_ctrl", TORCS_ctrl_Methods);
#endif
    import_array(); // Must be present for NumPy. Called first after above line.
#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}