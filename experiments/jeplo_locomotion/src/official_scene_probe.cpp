// Static geometry audit using the upstream TerrainSwitcher, no robot commands.
#include <mujoco/mujoco.h>
#include "terrain_switcher.h"
#include <iostream>
int main(int argc,char**argv){
    if(argc!=2)return 2;
    char error[1024];mjModel*m=mj_loadXML(argv[1],nullptr,error,sizeof(error));
    if(!m){std::cerr<<error;return 3;}
    mjData*d=mj_makeData(m);mj_forward(m,d);TerrainSwitcher sw;sw.update(m,d);
    for(char key: {'1','2'}){
        sw.queueKey(key);sw.update(m,d);
        std::cout<<"SCENE "<<key<<" timestep="<<m->opt.timestep<<" base="<<d->qpos[0]<<","<<d->qpos[1]<<","<<d->qpos[2]<<"\n";
        for(int g=0;g<m->ngeom;g++){
            const char*body=mj_id2name(m,mjOBJ_BODY,m->geom_bodyid[g]);
            const char*name=mj_id2name(m,mjOBJ_GEOM,g);
            if(!body||(!m->geom_contype[g]&&!m->geom_conaffinity[g]))continue;
            std::string b(body);if(b!="stairs"&&b!="boxes")continue;
            std::cout<<"GEOM "<<(name?name:"unnamed")<<" body="<<body<<" type="<<m->geom_type[g]<<" world_xyz="<<d->geom_xpos[3*g]<<","<<d->geom_xpos[3*g+1]<<","<<d->geom_xpos[3*g+2]<<" halfsize="<<m->geom_size[3*g]<<","<<m->geom_size[3*g+1]<<","<<m->geom_size[3*g+2]<<" friction="<<m->geom_friction[3*g]<<"\n";
        }
    }
    mj_deleteData(d);mj_deleteModel(m);
}
